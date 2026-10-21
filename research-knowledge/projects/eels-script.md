# EELS script

Project ID: `eels-script`. Code and workflow metadata share this Git repository
during the pilot. See the root README for the established scientific tools.

Existing applications: EELS Studio and Torched TACAW Studio. Their dependencies,
test suites and frozen run bundles remain separate from this bootstrap.
Shared runner consumption and immutable release manifests have not been
established. No site has an active workflow release.

The [BTO single-domain preparation](bto-single-domain.md) has scoped manual
Arrhenius validation records and reusable experiment code under
`research-workflow/experiments/bto-single-domain/`. These short checks establish
execution and trajectory handling for that experiment, not long-time domain
stability, spectral convergence, or an automatically deployable shared runner.
