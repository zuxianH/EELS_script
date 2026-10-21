"""Contract tests using disposable repositories; no services or cluster access."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_workflow import cli


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "project"
        self.root.mkdir()
        self.git("init", "-b", "main")
        self.git("config", "user.email", "test@example.invalid")
        self.git("config", "user.name", "Workflow test")
        source = Path(__file__).resolve().parents[1]
        shutil.copytree(source, self.root / "research-workflow",
                        ignore=shutil.ignore_patterns("__pycache__"))
        (self.root / ".gitignore").write_text(".rw-state/\n__pycache__/\n")
        (self.root / "science.py").write_text("temperature = 100\n")
        self.git("add", ".")
        self.git("commit", "-m", "Initial fixture")

    def git(self, *args, root=None):
        return cli.git(root or self.root, *args)

    def start(self, task="regression", agent="codex", **kwargs):
        return cli.start_task(self.root, task, "Fix lifecycle", "local", agent, **kwargs)

    def args(self, **values):
        options = dict(task="regression", summary="Implemented local change", next_action="Review result",
                       agent="codex", check=["unit tests: pass"], failed_attempt=[], job_id=[],
                       uncertainty=["Cluster not tested"], publish=False)
        options.update(values)
        return argparse.Namespace(**options)

    def test_overlapping_edits_are_isolated_and_original_is_preserved(self):
        first = self.start(agent="claude")
        second = self.start(agent="codex")
        (first / "science.py").write_text("temperature = 200\n")
        (second / "science.py").write_text("temperature = 300\n")
        self.assertEqual((self.root / "science.py").read_text(), "temperature = 100\n")
        self.assertNotEqual((first / "science.py").read_text(), (second / "science.py").read_text())
        self.assertIn("task/regression/claude", self.git("worktree", "list"))

    def test_dirty_source_is_not_silently_omitted_from_new_worktree(self):
        (self.root / "science.py").write_text("temperature = 200\n")
        with self.assertRaisesRegex(ValueError, "Commit or preserve"):
            self.start()
        self.assertEqual((self.root / "science.py").read_text(), "temperature = 200\n")

    def test_verified_record_requires_criterion_and_evidence(self):
        work = self.start()
        for extra in ({}, {"criterion": "Normal exit"}, {"evidence": ["exit.json"]}):
            with self.assertRaisesRegex(ValueError, "criterion and evidence"):
                cli.event(work, "regression", "Shutdown passed", "Inspect", status="verified", **extra)
        path, record = cli.event(work, "regression", "Shutdown passed", "Inspect",
                                 status="verified", criterion="Exit code 0", evidence=["test:local"])
        self.assertTrue(path.is_file())
        self.assertEqual(record["code_revision"], self.git("rev-parse", "HEAD", root=work))

    def test_events_are_append_only_and_context_filters_site_and_project(self):
        work = self.start()
        for index in range(20):
            cli.event(work, "regression", f"Observation {index}", "Continue")
        self.assertEqual(len(list((cli.knowledge(work) / "events").glob("*.json"))), 21)
        for project, site, name in (("eels-script", "local", "relevant"),
                                     ("eels-script", "dardel", "wrong-site"),
                                     ("other-project", "local", "wrong-project")):
            _, record = cli.event(work, "other-task", name, "Continue", site=site,
                                  status="verified", criterion="Test passed", evidence=["test:fixture"])
            record["project"] = project
            cli.save(cli.knowledge(work) / "lessons" / f"{name}.json", record)
        ctx = cli.context(work, "regression", "local")
        self.assertEqual(len(ctx["recent_events"]), 12)
        self.assertEqual([lesson["claim"] for lesson in ctx["scoped_verified_lessons"]], ["relevant"])
        digest = ctx["knowledge_digest"]
        cli.event(work, "regression", "New observation", "Continue")
        self.assertNotEqual(digest, cli.context(work, "regression", "local")["knowledge_digest"])

    def test_handover_rejects_uncommitted_code_and_existing_staging(self):
        work = self.start()
        (work / "science.py").write_text("temperature = 200\n")
        with self.assertRaisesRegex(ValueError, "Commit safe task code"):
            cli.handover(work, self.args())
        self.git("add", ".", root=work)
        with self.assertRaises(ValueError):
            cli.handover(work, self.args())

    def test_handover_contains_commit_failed_attempts_and_next_action(self):
        work = self.start()
        (work / "science.py").write_text("temperature = 200\n")
        self.git("add", ".", root=work)
        self.git("commit", "-m", "Task code and records", root=work)
        result = cli.handover(work, self.args(failed_attempt=["First attempt timed out"]))
        record = cli.read(Path(result["handover"]))
        self.assertEqual(record["code_revision"], self.git("rev-parse", "HEAD", root=work))
        self.assertIn("science.py", record["files_changed"])
        self.assertEqual(record["failed_attempts"], ["First attempt timed out"])
        self.assertFalse(result["published"])

    def test_cross_assistant_continuation_inherits_task_code_and_records(self):
        first = self.start(agent="claude")
        (first / "science.py").write_text("temperature = 200\n")
        self.git("add", ".", root=first)
        self.git("commit", "-m", "Claude checkpoint", root=first)
        result = cli.handover(first, self.args(agent="claude"))
        self.git("add", ".", root=first)
        self.git("commit", "-m", "Claude handover", root=first)
        second = self.start(agent="codex", base_ref="task/regression/claude")
        self.assertEqual((second / "science.py").read_text(), "temperature = 200\n")
        self.assertEqual(cli.context(second, "regression", "local")["last_handover"]["record_id"],
                         cli.read(Path(result["handover"]))["record_id"])
        self.assertIn("task/regression/codex", self.git("branch", "--show-current", root=second))

    def test_publish_to_bare_remote_and_resume_from_fresh_clone(self):
        remote = Path(self.temp.name) / "remote.git"
        subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
        self.git("remote", "add", "origin", str(remote))
        work = self.start()
        self.git("add", ".", root=work)
        self.git("commit", "-m", "Safe task checkpoint", root=work)
        result = cli.handover(work, self.args(publish=True))
        self.assertTrue(result["published"])
        remote_head = self.git("ls-remote", "origin", "refs/heads/task/regression/codex", root=work).split()[0]
        self.assertEqual(remote_head, self.git("rev-parse", "HEAD", root=work))
        clone = Path(self.temp.name) / "other-machine"
        subprocess.run(["git", "clone", "--branch", "task/regression/codex", str(remote), str(clone)],
                       check=True, capture_output=True)
        resumed = cli.start_task(clone, "regression", None, "local", "claude")
        self.assertIsNotNone(cli.context(resumed, "regression", "local")["last_handover"])

    def test_secret_record_blocks_publication(self):
        work = self.start()
        self.git("add", ".", root=work)
        self.git("commit", "-m", "Task checkpoint", root=work)
        with self.assertRaisesRegex(ValueError, "Possible secret"):
            cli.handover(work, self.args(summary="password=do-not-publish", publish=True))

    def test_identifiers_cannot_escape_knowledge_directory(self):
        for task in ("../escape", "/tmp/escape", "x/y", "--upload", ""):
            with self.assertRaises(ValueError):
                cli.identifier(task)

    def test_committed_credential_or_bulk_data_is_not_published(self):
        work = self.start()
        (work / "private.key").write_text("do not publish")
        self.git("add", ".", root=work)
        self.git("commit", "-m", "Accidental credential fixture", root=work)
        with self.assertRaisesRegex(ValueError, "Excluded data/credential"):
            cli.handover(work, self.args(publish=True))

    def test_deleting_credential_later_does_not_make_its_history_safe(self):
        work = self.start()
        (work / "private.key").write_text("do not publish")
        self.git("add", ".", root=work)
        self.git("commit", "-m", "Accidental credential fixture", root=work)
        self.git("rm", "private.key", root=work)
        self.git("commit", "-m", "Remove credential fixture", root=work)
        with self.assertRaisesRegex(ValueError, "Excluded data/credential"):
            cli.handover(work, self.args(publish=True))

    def test_worktree_session_lock_rejects_second_launcher(self):
        work = self.start()
        target = work / ".rw-state/agent.lock"
        with cli.lock(target):
            with self.assertRaisesRegex(ValueError, "Another process"):
                with cli.lock(target):
                    self.fail("Concurrent session acquired the same lock")

    def test_prepare_only_produces_revisioned_context_without_an_ai_call(self):
        output = io.StringIO()
        with redirect_stdout(output):
            result = cli.main(["--repo", str(self.root), "agent", "start", "--task", "prepared",
                               "--goal", "Prepare without launch", "--agent", "claude", "--prepare-only"])
        self.assertEqual(result, 0)
        state = json.loads(output.getvalue())
        package = cli.read(Path(state["context"]))
        self.assertEqual(package["code_revision"], state["code_revision"])
        self.assertEqual(package["knowledge_digest"], state["knowledge_digest"])
        self.assertIsNone(package["site_default"])

    def test_cli_status_exposes_disabled_cluster_automation(self):
        out = io.StringIO()
        with redirect_stdout(out):
            code = cli.main(["--repo", str(self.root), "status"])
        self.assertEqual(code, 0)
        status = json.loads(out.getvalue())
        self.assertFalse(status["automation"]["production_submission_enabled"])
        self.assertIsNone(status["sites"]["lumi"]["verification_budget"])


if __name__ == "__main__":
    unittest.main()
