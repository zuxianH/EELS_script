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
