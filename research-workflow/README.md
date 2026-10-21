# Unified research workflow

This is the local bootstrap of the supplied implementation brief, inside
EELS_script. It uses Python 3.11+ and Git, with no extra Python dependencies.
It does not replace either Studio runtime or any existing MD launch script.

## Current capability

- Separate branch/worktree for each task and assistant, with a lock against
  launching two assistant processes in the same managed worktree.
- Canonical `AGENTS.md`, explicitly imported by `CLAUDE.md`.
- Unique, append-only evidence records with status, scope, source revision,
  verification criterion, evidence references and next action.
- Compact task/site context, selected policies and scoped verified lessons,
  with the source Git revision, content digest and load timestamp.
- Code-backed handovers containing checks, failed attempts, uncertainty and job
  IDs. Optional ordinary Git publication preserves all existing remote history.
- Fetch-only synchronization that never edits the active checkout.

The [Phase A inventory](docs/inventory-2026-10-21.md) records the actual findings.
Both LUMI/Dardel SSH probes failed authentication. Site modules/containers in the Studio
are templates, not evidence of installed software or working MD launches.
No cluster submission, promotion, rollback, scheduler monitoring, background
coordinator, conflict-repair service, or production command is installed.
The configuration deliberately leaves all execution flags disabled and budgets
unset. Changing a flag cannot enable those unimplemented capabilities.

The subsequent [Arrhenius BTO experiment](experiments/bto-single-domain/README.md)
preserves the small minimization/NVT inputs, trajectory tools, environment lock,
and a portable manual submission template. Its [evidence record](docs/bto-single-domain-arrhenius-2026-10-21.json)
identifies two completed, bounded Slurm checks, including 25,000 atoms. This is
standalone LAMMPS evidence, separate from the pending LUMI/Dardel i-PI lifecycle
work. The adapted submission template has not itself been submitted. No generic
workflow release or production activation follows from these checks.

Resume the record-maintenance task with
`./rw context --task bto-single-domain --site arrhenius`. Its task identity is
`local` because it updates repository records; its physics evidence and scoped
lessons explicitly identify `arrhenius`.

## Start a task

Commit or otherwise preserve existing edits first. New worktrees contain only
committed files. Run commands from this repository or pass `./rw --repo PATH`.

```bash
./rw status
./rw task start fix-shutdown --goal 'Diagnose the idle allocation' --site dardel
```

The command prints a sibling path, normally
`../EELS_script-worktrees/fix-shutdown-codex`. Change to that directory before
editing or recording progress. Task records live in that worktree so its
commits carry both source and context. There is no shared writable source copy.

To launch an installed assistant, use:

```bash
./rw agent start --project eels-script --task fix-shutdown --site dardel \
  --agent codex --goal 'Diagnose the idle allocation'
```

Use `--agent claude` for Claude Code. No agent is started during setup.
`--prepare-only` creates/resumes the worktree and writes the context without
starting an assistant, including when that assistant is not installed.
Starting the launcher prints the machine, task branch, code revision, knowledge
revision/digest and site default. A null default means no validated release.
Claude was not installed on this machine during initial inventory; its actual
startup/import behavior has not been demonstrated here.

The launcher passes the context file path in the assistant's initial prompt.
The records prove which package was prepared, not that the model read it.
It writes start/exit events when those points are reached. AGENTS.md asks the
assistant to checkpoint meaningful progress; there are no unverified lifecycle
hooks or unattended automatic code commits. A crash can lose recent reasoning.

## Checkpoint and prepare a handover

Inside the task worktree:

```bash
./rw task checkpoint fix-shutdown \
  --claim 'Client remains alive after the server exits' \
  --status observed --scope 'This Dardel test run only' \
  --evidence 'run-id:client-lifecycle-check' \
  --next-action 'Inspect the owned process group and batch step exit records'

./rw context --task fix-shutdown --site dardel
```

Use `hypothesis` for an explanation not yet established. `verified` requires
both `--criterion` and at least one `--evidence`. This checks the record's
structure; it cannot establish the truth of arbitrary evidence. Neither
agreement between models nor this label authorizes promotion. Raw logs remain
at their original location; record small sanitized findings and references.

Review and commit only the intended small code/configuration/record files with
Git. Do not run a blanket `git add .` in a research directory containing data.
Then prepare a handover:

```bash
./rw task handover fix-shutdown \
  --summary 'Current diagnosis and implemented changes' \
  --check 'Local contract tests passed; evidence: test-run-ID' \
  --failed-attempt 'Describe any unsuccessful approach' \
  --uncertainty 'Cluster regression remains untested' \
  --next-action 'Collect the missing cluster evidence'
```

The command refuses uncommitted source and preserves an existing staging area.
It creates a unique handover and event instead of rewriting a shared log.
Review these records, then commit them, or repeat with `--publish` to commit
only this task's new metadata/events/handover and push the current task branch.
Unrelated records and files are never automatically staged. Code already
committed on that branch is included in its push, so review that history too.
Publication rejects common data/credential paths, symlinks and changed files
larger than 1 MiB. Secret detection on records is conservative, not a guarantee
of sanitization. Small source files still require review for embedded secrets.

A failed push leaves the local checkpoint intact and exits nonzero. Configure
GitHub authentication on the machine before relying on publication. No tokens
are embedded in remotes, configuration or records. `--publish` never force-pushes
and does not merge the task into main or activate a release.

## Continue with another tool or machine

Push the task branch and its handover first. On another authenticated machine,
clone this repository if needed and fetch the published branch:

```bash
./rw sync
./rw agent start --project eels-script --task fix-shutdown --site dardel \
  --agent claude --from origin/task/fix-shutdown/codex --prepare-only
```

This creates a Claude worktree from the published Codex code and records.
Inspect the generated context, then repeat without `--prepare-only` to launch
an installed Claude CLI. The reverse direction works with the same mechanism.
If a worktree already exists for this task/assistant, the launcher resumes it
without changing its files. It never silently rebases an existing session onto
a newly fetched handover. Integrate new changes separately and retest them.

Use a different task ID for another concurrent session of the same assistant.
Refreshing context is explicit: run `rw context` again and have the assistant
read the new result at a checkpoint. Fetching files does not refresh a running
model's context. Full remote assistant-to-assistant execution remains unverified;
tests cover the transport and context contract in disposable clones only.

## Knowledge organization and future extraction

`research-knowledge/` is initially versioned alongside this project so context
and code can be published atomically. Its policy, lesson, task, event, project
and site directories follow the brief's independent knowledge-store layout.
Records should be added, superseded explicitly, and indexed from source rather
than merging every agent into one mutable log. Lessons enter context only if
their explicit project and site match; inspect their version/engine conditions
in `scope` before applying them. Hypotheses are never imported as policies.

This is not yet two separate shared GitHub repositories. After the pilot, extract
the generic code into `research-workflow` and knowledge into `research-knowledge`,
retain history, and pin this project's dependency using exact release manifests.
Do not create editable runner forks per cluster. Cross-project registration,
manifest consumption and revalidation are pending that stage.

## Remaining implementation sequence

1. Complete authenticated Phase A comparison using actual MD scripts, tiny
   inputs, executable/build fingerprints, process trees, logs and scheduler
   evidence. The allocation-persistence root cause remains unknown.
2. Establish explicit site budgets and extract the shared runner. Test normal
   completion, early server exit, failed/stuck clients and interrupted supervision
   locally, then on both clusters. Cleanup must be restricted to owned process
   groups or job/step IDs. Test physics independently using justified references.
3. Demonstrate real cross-assistant/cross-machine resumption with a published
   source commit and record the context actually read by each assistant.
4. Build the serialized coordinator with a persistent external ledger, submission
   reconciliation, bounded repair attempts and per-project/site validation.
   Protect gates outside candidate control. Use exact immutable manifests and
   compare-and-swap promotion; failure retains each site's last good release.
5. Freeze release/inputs/configuration for every submitted job. Promotion or
   rollback affects future jobs only. Production requires a separate explicit
   request identifying release, inputs and resources.

No current command submits a job, so reconnecting this bootstrap cannot create
duplicate allocations. Reconnect/submission behavior of the future coordinator
has not been implemented or validated.

## Verification and documentation

```bash
python3 -m unittest discover -s research-workflow/tests -v
git diff --check
```

Tests use disposable repositories and bare remotes. They verify overlapping
edits remain isolated, dirty source is preserved, incomplete handovers fail,
knowledge is scoped, record digests change, and published task commits can be
resumed in a fresh clone. They do not invoke AI services or cluster commands.

Checked against Codex CLI 0.160.0 and the official documentation on 21 October
2026: [Codex instructions](https://developers.openai.com/codex/agent-configuration/agents-md)
and [Claude memory/imports](https://code.claude.com/docs/en/memory).
The explicit `@AGENTS.md` import follows the user's compatibility requirement.
Do not bypass either tool's normal approval, sandbox or authentication settings.
