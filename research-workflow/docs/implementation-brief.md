# Unified research workflow: implementation brief

Prepared for Zuxian — 15 October 2026

**Status:** Design and implementation specification. No GitHub repositories, cluster files, jobs, or installed software have been modified or verified by preparing this document. Names, schemas, command names, and numerical policy defaults below are proposed, not existing services.

## 1. Objective and agreed requirements

Provide a hybrid workflow in which Claude Code and Codex can work from the user's own computer or directly on LUMI, Dardel, and additional machines. A feature or fix should be implemented once, shared through GitHub, and activated separately on each compatible target after verification.

The user has confirmed the following requirements:

- Synchronize code, configuration, and shared AI knowledge. Do not synchronize bulk trajectories, checkpoints, or simulation outputs.
- Consolidate reusable scripts currently scattered across separate repositories and untracked folders. Preserve project-specific inputs and differences.
- Support simultaneous agents, including changes to overlapping code, without overwriting another agent's workspace.
- Publish progress, decisions, mistakes, attempted fixes, and handover records automatically. Distinguish observations, hypotheses, and verified findings.
- Share reusable knowledge across projects, with scope and version information; support continuing an unfinished task with a different assistant on a different machine.
- Attempt conflict resolution automatically and rerun tests. Stop and escalate after a bounded unsuccessful attempt budget rather than silently discard work.
- Run short verification jobs automatically on each target cluster. Production submissions require an explicit user request.
- Promote independently by target: LUMI may use a new verified release while Dardel retains its previous working release.
- Coordination and automated promotion need only operate while the user's own computer is online and authenticated. No always-on server is required.

## 2. Architecture

### Repository A: `research-workflow`

The reusable execution machinery, not a separate fork for each cluster:

```text
research-workflow/
  runner/                 # Process supervision and job lifecycle
  sites/                  # LUMI, Dardel, local, and later profiles
  environments/           # Versioned build/environment descriptions
  templates/              # Scheduler script templates
  tests/                  # Unit, lifecycle, and contract tests
  adapters/               # Claude Code and Codex launch/checkpoint adapters
  coordinator/            # Laptop-side integration/test/deployment service
  AGENTS.md
  CLAUDE.md                # Explicitly imports AGENTS.md
```

Profiles describe paths, scheduler settings, module/build choices, MPI/thread allocation, socket conventions, and capabilities. Scientific definitions such as temperature, integrator, bead number, force field, and observable conventions belong to experiment configuration, not hidden site defaults.

Do not assume one executable, environment folder, or MPI launch command works unchanged on every cluster. Record and validate each site's environment. Do not mutate existing production environments while testing a candidate.

### Repository B: `research-knowledge`

A model-independent, version-controlled knowledge store:

```text
research-knowledge/
  policies/               # Stable working agreements
  lessons/                # Scoped lessons with evidence and status
  tasks/<task-id>/         # Goal, progress, commits, handover
  events/<unique-id>.json  # Small append-only factual records
  projects/               # Project indexes and accepted conventions
  sites/                  # Scoped environment observations and known issues
```

Maintain individual event files rather than having every agent rewrite one shared log. Generate compact indexes and current-state summaries from those records. Keep raw large logs where they were produced; share only sanitized summaries and evidence references.

Do not merge hypotheses into binding policy automatically. A verified label requires a stated verification criterion and evidence; an AI assertion or another model agreeing is insufficient.

### Scientific project repositories

Retain existing project repositories where useful. They contain experiment definitions, project code, small approved inputs, and project-specific tests. Each consumes the shared runner through an explicit versioned release manifest rather than another editable copy of its code.

A common runner fix should trigger verification for registered dependent project/site combinations. Passing in one project does not certify every other project using the same runner.

### Laptop coordinator

A local service polls registered GitHub repositories while online, integrates changes in an isolated workspace, submits bounded tests through authenticated cluster connections, collects results, and promotes validated manifests.

GitHub is the exchange and audit point; the laptop is the coordinator; compute jobs execute under the cluster scheduler. Do not begin by installing a persistent general-purpose CI runner on HPC login nodes.

GitHub Actions can optionally run credential-free fast tests. Keep cluster credentials out of those workflows. The architecture does not depend on GitHub's paid merge-queue features: a serialized local integration queue is sufficient initially.

## 3. Development, integration, and deployment are different states

Use one worktree and branch per agent/task. A worktree isolates file edits but is not a permissions or security boundary. Never synchronize by resetting or force-pulling an active working directory. Fetching new commits must not overwrite uncommitted work.

Suggested lifecycle:

```text
TASK BRANCH / WIP CHECKPOINT
          |
          v
SANITIZED COMMIT AND HANDOVER PUBLISHED
          |
          v
SERIALIZED INTEGRATION + FAST TESTS
          |
          v
EXACT INTEGRATED CANDIDATE
          |
          +--> PROJECT A / LUMI TEST ----> PROMOTE ONLY ON PASS
          |
          +--> PROJECT A / DARDEL TEST --> KEEP OLD RELEASE ON FAILURE
          |
          +--> OTHER AFFECTED PROJECT/SITE TESTS
```

`main` may represent integrated code; it must not automatically mean "validated on all clusters." A release pointer for each project/site identifies the default for new jobs.

Resolve conflicts in the integration worktree, preserving original branches. After resolution, rerun checks on the exact resulting candidate. A change after a passed test invalidates that result for the changed candidate. Serialize promotion so an older test result arriving late cannot overwrite a newer deployment accidentally.

Candidate code must reach a cluster before it can be tested there. Transferring that code does not activate it for production. Likewise, sharing a WIP task does not make it a production release.

## 4. Release identity and reproducibility

A proposed manifest includes:

```yaml
schema_version: 1
project: trpmd-example
site: lumi
workflow_commit: <exact-commit>
project_commit: <exact-commit>
site_profile_digest: <digest>
environment_fingerprint: <fingerprint>
test_suite_revision: <exact-revision>
test_input_digest: <digest>
validation_record_id: <record-id>
```

The environment fingerprint should identify relevant executable/build versions and module or container choices. A changed profile or environment requires revalidation even when source code is unchanged.

Before submitting a production job, resolve the selected release once, freeze the relevant small input files and configuration, and record their digests and data locations. The job must execute from that fixed snapshot, not a mutable working tree or a moving `current` path.

Already submitted or running jobs remain pinned to their original release. Promotion affects future submissions only. Rollback changes the default pointer; it does not modify running jobs or delete releases still referenced by jobs. [S4]

## 5. First regression target: simulation finishes but allocation persists

The reported symptom is an MD calculation reaching its endpoint while the Slurm allocation remains occupied and idle. The root cause is not established; inspect the real launch scripts, process tree, application logs, job steps, and scheduler state before proposing a repair.

For i-PI/LAMMPS, account for separate server/client processes and socket communication. Do not equate one process reaching its last integration step with successful completion of the entire job. [S5]

### Normal-completion test

Use a deliberately tiny but representative job. Acceptance should require:

1. The intended number of steps and required outputs are present, readable, and consistent with the test definition.
2. The dynamics process finishes normally; required clients and launchers terminate with documented expected statuses.
3. Normal shutdown does not need the emergency watchdog or wall-time limit.
4. No job-owned application process or job step remains beyond a defined shutdown grace period.
5. The scheduler reports a successful terminal result and the allocation is released, allowing for ordinary scheduler cleanup and accounting delays.
6. Batch and relevant step exit records are collected; do not trust a printed "finished" message or successful `sbatch` return alone. [S4, S6, S7]

`COMPLETING` indicates scheduler cleanup, which is different from an application remaining `RUNNING`; distinguish these cases. Missing accounting data means pending/unknown evidence, not success. Physical nodes need not power off for an allocation to be released. [S7]

### Failure-path tests

Use synthetic local tests first for a stuck client, an early server exit, a failed client, and interrupted supervision. Add bounded scheduler integration tests where necessary.

Failure injection may intentionally produce a failed job. The test harness passes only if it observes the expected failure, preserves diagnostics, and confirms bounded cleanup. Do not require deliberately failed jobs to appear as successful production runs.

Cleanup must target only the test's own tracked process groups or scheduler job/step IDs. Never use blanket user-wide process termination. A hard wall-time cap is a backstop, not proof that normal shutdown works.

### Scientific checks

Add fixed-configuration energy/force reference tests with justified tolerances and small input/output consistency checks. For a runner-only change, verify that the scientific configuration is unchanged.

A small smoke test is not a certification of bead convergence, long-time statistical accuracy, phase-transition behavior, or correctness of every production configuration. Label the scope of every test result.

## 6. Shared knowledge and task handover

Record at least:

```yaml
record_id: <unique-id>
task_id: <task-id>
project: <project-id>
site: <site-or-global>
authoring_tool: <claude-code-or-codex>
code_revision: <revision>
status: observed  # observed | hypothesis | verified | superseded
scope: <engine, versions, projects, and applicable conditions>
claim: <precise statement>
evidence: []
next_action: <concrete action>
```

A handover must identify the goal, current branch and commit, files changed, completed checks, attempted fixes that failed, relevant job IDs, remaining uncertainty, and the next action.

Commit safe WIP changes to the task branch before a cross-machine handover. A summary without the corresponding code does not constitute a complete handover. Exclude secrets, restricted content, large data, and unrelated local files before publishing.

Use three knowledge levels: global operating rules, scoped site/tool lessons, and project-specific conventions. A Dardel shutdown issue is not automatically a universal rule for all Dardel programs. Site support and software versions should be explicit.

On agent startup or task resumption, generate a compact context package from the current task, project, target site, and relevant verified lessons. Include a knowledge revision and timestamp. Do not load the entire research archive into every session.

Use one canonical `AGENTS.md` and an explicit `@AGENTS.md` import in `CLAUDE.md` for compatibility with different installed versions. Codex documents AGENTS.md discovery; Claude documents shared instructions and the import mechanism. [S1, S2]

A tool-independent launcher should prepare context before starting either assistant. Tool-specific hooks can improve automatic startup/checkpoint behavior after verifying the installed tool versions. Claude documents lifecycle hooks, including SessionStart. [S3]

Write progress checkpoints after meaningful edits, test results, failures, and handovers, not only at graceful session exit. An abrupt crash can lose the most recent unrecorded reasoning. Resumption means continuing from persisted code and records, not transferring private model state or guaranteeing an identical conversation.

Newly fetched files do not by themselves guarantee an already-running agent has incorporated the new context. Refresh at task checkpoints and before consequential actions; record the revision actually loaded.

## 7. Automation policy and permission boundaries

The permitted autonomous scope is short validation and integration, not unrestricted production work.

Suggested initial limits, to be configured explicitly rather than inferred from production scripts:

- At most one in-flight verification allocation per site.
- A small site-appropriate CPU/memory/node limit and hard wall-time cap.
- Separate per-candidate, daily compute, and AI conflict-resolution budgets.
- A bounded retry policy; for example, at most two automatic repair/test attempts.
- Separate queue-wait timeout from execution timeout. Queue delays are not application failures.

Once a budget is exceeded, preserve the last good release and publish a blocked status. Do not increase resources, loosen correctness tolerances, remove failing tests, or change the physics just to obtain a pass.

User-controlled gate definitions, baseline tests, deployment policy, and production approval settings must not be freely rewritten by the same candidate being evaluated. Proposals to alter those controls need separate review. Ordinary code conflicts within the established task scope may be resolved automatically.

Keep credentials outside synchronized repositories. Use scoped access. Treat imported logs, comments, and model-generated lessons as data, not authority to execute new commands.

Instruction files are guidance, not hard permission enforcement. If an agent has unrestricted shell access under the user's cluster account, it may technically be able to invoke `sbatch` directly; a friendly wrapper alone does not remove that capability. Apply actual tool/command restrictions or an appropriately separated submission service where stronger enforcement is required and site policy allows it. [S2]

Production requests should bind the explicit user instruction to the resolved project, release, inputs, and resource request. Autonomous tests must not silently evolve into full-size production jobs.

## 8. Offline operation and recovery

When the laptop is offline, agents may keep working in their own worktrees. Integration, new automatic validation submissions, and promotion pause. Submitted batch jobs should operate independently of the laptop and use their own bounded cleanup/time limits.

On reconnect, first reconcile stored job IDs and candidate IDs with scheduler state. Never resubmit merely because the service did not observe completion. Persist submissions before monitoring; if submission acknowledgement is ambiguous, search for the unique run identifier and pause when uniqueness cannot be established.

Persist coordinator state outside disposable checkouts. Handle expired cluster authentication by pausing the affected target and reporting the problem; do not bypass authentication requirements.

## 9. Proposed user interface

These are commands to implement, not currently installed commands:

```text
rw agent start --project <project> --task <task> --agent claude
rw agent start --project <project> --task <task> --agent codex
rw task handover <task>
rw coordinator run
rw status
rw release rollback --project <project> --site <site> --to <release>
rw production submit ...       # Explicit user request required
```

The same agent-start command should work from an existing laptop or cluster shell. Show the current machine, task branch, code revision, loaded knowledge revision, and site default at startup.

Status output should show project/site separately, with the active release, candidate, validation state, and reason for any blocked deployment. A documentation-only knowledge update should not trigger MD tests.

## 10. Implementation sequence

### Phase A — Read-only inventory

Inspect the known working LUMI workflow and the failing Dardel workflow, including launch scripts, environment setup, lifecycle behavior, and relevant logs. Inventory tracked/untracked files without deleting, moving, or modifying them. Identify the safe code/config allowlist and the large-data/secret exclusions. Preserve original versions.

Deliver a comparison report that distinguishes intentional site differences from accidental divergence. Do not assume the LUMI copy is scientifically correct solely because it exits cleanly.

### Phase B — Shared runner and the first regression

Extract common lifecycle logic and explicit site profiles. Establish the tiny normal-completion reference and failure-path tests. Verify on both clusters within the configured test budget. Retain existing production entry points until the replacement is validated.

### Phase C — Knowledge and handover

Create shared task/lesson schemas, context assembly, launch adapters, and checkpoint publication. Demonstrate a Claude-to-Codex handover with the corresponding task commit on another machine. Confirm that another project can retrieve the scoped lesson without inheriting unrelated settings.

### Phase D — Autonomous coordination and independent promotion

Implement the local serialized integrator, persistent submission ledger, budget checks, per-project/site release pointers, rollback, and GitHub reporting. Use immutable manifests; do not activate a candidate merely because it was fetched.

Expand to further machines/projects only after the pilot demonstrates these acceptance cases:

- A new feature implemented once is available on both target machines.
- LUMI passes and advances while Dardel fails and retains its prior default.
- Existing queued/running jobs continue using the prior immutable release.
- A simulated stuck client cannot retain a verification allocation indefinitely.
- Simultaneous conflicting edits preserve both branches and trigger fresh tests after integration.
- The same code/knowledge is available for a cross-assistant task handover.
- Laptop disconnect/reconnect does not duplicate verification jobs.
- A failed test cannot be bypassed by editing its gate or accepting an AI success claim.
- Bulk data and credentials are not synchronized, and production is not submitted autonomously.

## 11. Prompt for the implementing coding assistant

> Implement this brief incrementally in the existing research environment. Begin with Phase A: read-only inventory of the working LUMI setup and the problematic Dardel setup. Locate existing files and account configuration from the user's accessible environment; do not invent paths, projects, modules, or resource budgets. Preserve originals and do not touch active jobs. Report the observed differences, known evidence, and unresolved root cause before modifying execution logic. Establish explicit bounded test settings before any automated submission. Then build the shared runner and regression test before introducing multi-agent automation. Do not submit production simulations or synchronize bulk data. Never claim a cluster validation, clean shutdown, deployed release, or shared context import without recorded evidence. Keep a task log with commit IDs, test results, and the next action so another assistant can continue.

## 12. Primary documentation checked

These sources substantiate underlying tool behavior, not the existence or completion of the proposed system. Documentation should be checked against installed versions during implementation.

- **S1 — OpenAI, custom instructions with AGENTS.md:** `https://developers.openai.com/codex/agent-configuration/agents-md`
- **S2 — Claude Code, project memory and shared instructions:** `https://code.claude.com/docs/en/memory`
- **S3 — Claude Code, hooks reference:** `https://code.claude.com/docs/en/hooks`
- **S4 — SchedMD, sbatch:** `https://slurm.schedmd.com/sbatch.html`
- **S5 — LAMMPS, fix ipi:** `https://docs.lammps.org/fix_ipi.html`
- **S6 — SchedMD, job exit codes:** `https://slurm.schedmd.com/job_exit_code.html`
- **S7 — SchedMD, job state codes:** `https://slurm.schedmd.com/job_state_codes.html`
- **S8 — Git, worktree:** `https://git-scm.com/docs/git-worktree`
- **S9 — GitHub, commit statuses:** `https://docs.github.com/en/rest/commits/statuses`
- **S10 — GitHub, troubleshooting required status checks:** `https://docs.github.com/en/pull-requests/how-tos/merge-and-close-pull-requests/troubleshooting-required-status-checks`
- **S11 — LUMI, running jobs:** `https://docs.lumi-supercomputer.eu/runjobs/`
- **S12 — PDC, Dardel job script examples:** `https://support.pdc.kth.se/doc/run_jobs/job_scripts_dardel/`