# Research knowledge bootstrap

These small records travel with this project's commits. No independent global
knowledge repository or cross-project deployment is claimed yet.

| Directory | Contents |
| --- | --- |
| `policies/` | Stable user-controlled agreements. Propose policy changes separately from candidate evaluation. |
| `lessons/` | Scoped JSON claims using `schemas/event.schema.json`; never promote hypotheses automatically. |
| `tasks/<id>/` | Goal/baseline metadata and unique handover JSON files. |
| `events/` | Unique append-only factual records; `rw context` derives a compact task view. |
| `projects/` | Project indexes and accepted conventions. |
| `sites/` | Scoped site observations; no inferred scientific or resource defaults. |
| `schemas/` | Machine-readable event contract and example task metadata. |

The CLI validates required event fields and verification criterion/evidence.
It does not execute imported text or treat a record as a passed release gate.
Avoid secrets, private account information, absolute personal paths and raw logs
in records intended for publication. Keep large evidence where it was produced.

The [BTO single-domain task](tasks/bto-single-domain/README.md) records the
Arrhenius minimization/NVT preparation, bounded job results, setup failures and
remaining scientific checks. See [Arrhenius observations](sites/arrhenius.md)
and the project's [BTO knowledge note](projects/bto-single-domain.md).
