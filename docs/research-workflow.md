# Using the shared research workflow

The repositories are independent siblings:

```text
Documents/
  EELS_script/         scientific code and experiments
  research-workflow/   generic launcher and tests
  research-knowledge/  scoped records and handovers
```

The two shared repositories are private. On another machine clone all three
with your normal authenticated Git access. `.research-workflow.json` pins the
shared tool commit and configures both sibling paths. If workflow HEAD differs,
check out the pinned commit before using the launcher. Absolute or custom local
paths can be set through RW_WORKFLOW_PATH and RW_KNOWLEDGE_PATH; the code pin
still applies. The same resolution works in task worktrees.

From EELS_script:

```bash
./rw status
./rw context --task bto-long-nvt --site arrhenius
./rw agent start --task new-analysis --site local --agent codex   --goal 'Describe the analysis task'
```

The agent command starts an installed assistant in an isolated source worktree;
`--prepare-only` prepares context without launching it. Site flags identify
context and do not open SSH or submit jobs.

Tasks and handovers live in research-knowledge/tasks/eels-script/<task>/; event
UUIDs and explicit project/site scope keep concurrent records separate. Context
reports the actual source, workflow and knowledge revisions, plus a content
digest and whether knowledge has unpublished local edits.

Commit reviewed scientific source before a handover. `rw task handover ...
--publish` pushes the source task branch first, then commits/pushes only this
task's records in research-knowledge. These are separate Git transactions;
partial failures preserve a receipt and all records. Review both histories and
configure ordinary Git authentication before relying on command-line pushes.
`rw sync` fetches all three repositories without changing active working trees.
Update a clean knowledge branch explicitly and start a new code worktree from
the corresponding fetched task branch to resume on another machine.

BTO experiment source moved to experiments/bto-single-domain/. Historical
validation and job evidence moved to docs/workflow-history/. Scientific source
files and historical JSON records retain their original content. The shared
knowledge migration map resolves old embedded paths; the complete original
history remains reachable in this repository. See repository-split-2026-10-22.json
for provenance and file integrity information.

No scheduler automation, background coordinator, automatic startup or production
activation was added. Existing frozen jobs and external simulation data are
independent of these folder moves.

See the [shared tool guide](https://github.com/zuxianH/research-workflow) and
[knowledge repository](https://github.com/zuxianH/research-knowledge) for commands.

## Research skills

The shared repository now includes research-task and research-handover skills.
Their canonical files stay in research-workflow/skills/. On each computer, run
`python3 ../research-workflow/scripts/install_skills.py --target both` to create
non-overwriting personal discovery links for Codex and local Claude Code.
Use --target codex or --target claude for one assistant.

From this configured project, in Codex:

```text
$research-task Resume my BTO analysis on Arrhenius.
$research-handover Prepare a local handover for this task.
```

In local Claude Code use /research-task or /research-handover. Refresh or restart
an existing session if its skill picker has not refreshed. Personal local skills
need separate installation for cloud sessions and other machines.

Task preparation uses prepare-only mode for the current assistant; it does not
launch another assistant. A local handover is published only when requested or
already authorized. Publication verifies source and knowledge refs separately;
the immutable handover record's publication text is creation-time information,
while receipts and actual remote refs establish its current availability.
