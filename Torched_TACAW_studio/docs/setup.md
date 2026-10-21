# Laptop and cluster setup

The browser interface and calculation runtime can use separate Python environments.
Run the interface on the computer that contains the input files.
On a cluster, use the browser through an SSH tunnel.

## Laptop: interface

Use Python 3.11 or later.
On Linux or macOS:

```bash
cd /path/to/Torched_TACAW_studio
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
./run_app.sh
```

Open http://localhost:8502.
The shell launchers use Bash.
On Windows, use WSL for the complete local workflow.

## Laptop: calculation runtime

Create a separate environment:

```bash
python3 -m venv .runtime
.runtime/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.runtime/bin/python -m pip install -r requirements-runtime.txt
.runtime/bin/python -m pip install 'pyms @ git+https://github.com/HamishGBrown/py_multislice.git'
.runtime/bin/python -m pip install --no-deps -e /path/to/torched-tacaw-main
```

On macOS, use `pip install torch` if the CPU wheel index has no compatible wheel.
Select **Local CPU** in the interface.
Set **Simulation Python command** to `[/absolute/path/to/.runtime/bin/python]`.
Set **Python dependency paths** to `[/absolute/path/to/torched-tacaw-main]`, or leave it empty after an editable engine installation.

Use a copy of the supplied engine when you need its multi-bead functionality.
Upstream versions may have a different input interface.
The adapter targets the supplied 0.3.1 interface.

For NVIDIA GPUs, first install the Torch build appropriate for your driver and CUDA runtime.
For AMD GPUs, first install a supported ROCm Torch build.
Use [PyTorch's installation selector](https://pytorch.org/get-started/locally/) to select the platform-specific installation command.
Then install the remaining runtime requirements and engine with `--no-deps` so pip does not replace the Torch build.
Choose the corresponding Local CUDA or Local ROCm profile.
CPU is the default on Apple laptops.
MPS execution has not been validated for this engine.

On the current workspace, an existing runtime is available at:

```text
/scratch/project_465002371/zuxian/venvs/deepmd-ipi/bin/python
```

Use it with the engine source directory on PYTHONPATH.
The interface exposes this detected environment as **Workspace CPU**.
This does not install anything into that existing environment.

## Cluster: browser interface

Install the interface in a normal Python environment with network access.
Its NumPy, ASE, Plotly, and Streamlit packages do not require a GPU.
Run it on a login or interactive node according to the site's interactive-app policy.

```bash
./run_app.sh --server.address 127.0.0.1 --server.port 8502
```

From your laptop, forward that port:

```bash
ssh -N -L 8502:127.0.0.1:8502 your-cluster-login-alias
```

Open http://localhost:8502 on the laptop.
Use an SSH configuration alias containing your site's required authentication.
If the interface runs on another node, use the site's SSH jump-host procedure.
All paths entered in the interface refer to files on the machine serving it.

The interface does not implement remote file transfers or laptop-to-cluster submission.
A copied Studio YAML file can be loaded on another computer after its paths and profile are adjusted.

## Generic Slurm setup

Select **Generic Slurm**.
Set your project account, partition, time, memory, nodes, and CPUs.
Set exactly one task per GPU for GPU calculations.
CPU jobs do not emit GPU requests.

Machine profiles contain:

| Field | Meaning |
| --- | --- |
| `scheduler` | `local` or `slurm` |
| `backend` | `cpu`, `cuda`, or `rocm` |
| `python` | Python executable and optional arguments, as a list |
| `launcher` | Container command prefix, as a list |
| `modules` | Module names loaded on the host |
| `pythonpath` | Engine/dependency directories available to the calculation Python |
| `environment` | Explicit string environment values |
| `setup_commands` | Additional host shell setup |
| `worker_binding` | Device selection strategy |
| `srun_args` | Site-specific CPU/GPU binding arguments |

Use the site's current module names.
All compute nodes must see the run folder and trajectory files through shared storage.
Containers must bind every source, output, and dependency directory they need.
Use plain absolute paths unless an entire launcher token is an explicit environment reference such as `${SIFPYTORCH}`.
Profile paths are quoted literally.
Setup commands are explicit shell code.

**scheduler** binding requests `--gpus-per-task=1 --gpu-bind=single:1`.
It requires one GPU to be visible to each task.
**local-rank** uses `cuda:SLURM_LOCALID` when the whole node's GPUs are visible.
**rocr-local-rank** clears competing GPU visibility variables and sets `ROCR_VISIBLE_DEVICES=SLURM_LOCALID`.
Use that strategy only for a whole-node or equivalent allocation whose physical GPU numbering matches local ranks.
For shared/subset GPU allocations, use scheduler-controlled visibility.

## LUMI

The LUMI profile starts from the modules in the supplied examples:

```text
LUMI/25.03
partition/G
PyTorch/20240918-rocm-6.2.1-python-3.12-singularity-20241007
```

Verify them with `module spider` before use.
The template uses `singularity exec --rocm "${SIFPYTORCH}"`.
Use the site's recommended command if its container module uses different GPU integration.
See [LUMI PyTorch guidance](https://docs.lumi-supercomputer.eu/software/packages/pytorch/) and [GPU binding guidance](https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/distribution-binding/).

Set your account and partition in SBATCH.
A full GPU node may require eight GPU tasks, as in the BTO example.
Add the appropriate CPU-binding masks through **Additional srun arguments** when needed.
For example, keep a complete `--cpu-bind=mask_cpu:...` expression as one list entry.

Do not reuse the vanished `path_integral_MD/run_Si_TPRMD/TACAW/*` paths from the older scripts.
Provide current directories containing the engine and pyms dependencies.
Check every bind path before adding it to `SINGULARITY_BIND`.

## Dardel

The editable template follows [PDC's PyTorch container instructions](https://support.pdc.kth.se/doc/applications/pytorch/).
The documented starting point is:

```bash
module load PDC singularity/4.2.0-cpeGNU-24.11
singularity exec --rocm -B /cfs/klemming \
  /pdc/software/resources/sing_hub/rocm6.3_ubuntu24.04_py3.12_pytorch_release_2.4.0 \
  python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

Run GPU checks within an allocated GPU node.
List modules and container files on Dardel to confirm the current names.
PDC notes that examples can lag available versions.
Use [PDC's software list](https://support.pdc.kth.se/doc/applications/) and [job-script guidance](https://support.pdc.kth.se/doc/run_jobs/job_scripts/) to select current resources.
Your account and partition are deliberately not supplied by the template.

PDC containers cannot be modified in place.
Install extra dependencies into a project directory using the same Python version as the container.
Add that directory and your engine/pyms source directories to the profile's PYTHONPATH.
Bind the project paths into the container.

Example structure:

```text
/cfs/klemming/projects/<project>/tacaw/
    Torched_TACAW_studio/
    torched-tacaw-main/
    py_multislice/
    runtime_deps_py312/
    inputs/
```

From the configured container, install a dependency layer using `python3 -m pip install --target /path/to/runtime_deps_py312 ...`.
Check the container's package versions before selecting the missing packages.
Keep the container's Torch build.
Ensure NumPy/SciPy/ASE/Zarr satisfy the supplied engine's requirements.
Avoid blindly following older NumPy 1.x examples: this engine declares NumPy 2.1 or later.
Run **Check environment** after configuring the dependency layer.

## Cluster smoke tests

Use these checks before a production allocation:

1. Select a tiny trajectory with a valid fixed orthorhombic cell and PBC.
2. Choose plane wave, one worker, a small grid, a short chunk, and no initial skipped frames.
3. Generate a new run folder.
4. Submit **Check environment** and inspect `environment.json` and Slurm logs.
5. Submit the calculation. Confirm every batch finishes.
6. Convert the result to NumPy. Compare shape and values with the Zarr array.
7. Repeat with a three-point line and a 2×2 scan. Confirm their stored coordinates match the preview.
8. Use a separate tiny run to check cancellation and resume.
9. For multiple GPUs, confirm each worker log reports the intended device.
10. For multiple nodes, confirm every node can access the same inputs, package snapshot, locks, and result store.

No production jobs are submitted by the test suite.
The optional CPU smoke test uses two Si atoms, twelve frames, and very small arrays:

```bash
TACAW_SMOKE_PYTHON=/absolute/path/to/simulation/python \
TACAW_SMOKE_ENGINE=/absolute/path/to/torched-tacaw-main \
.venv/bin/python -m pytest -q tests/test_runtime_smoke.py
```

Cold imports on shared filesystems can be slow.
Set `TACAW_SMOKE_TIMEOUT` to change the 360-second timeout for each launch.
Timed-out tests terminate their complete worker process group.

## Troubleshooting

**The app opens but Torch cannot be imported:** Set the simulation Python command or container launcher.
The interface environment intentionally does not install Torch.

**No GPU is visible:** Check your allocation, Torch backend, launcher GPU flag, and task binding.
ROCm Torch uses `cuda:N` device names.
Do not run GPU environment checks on the login node.

**A module command is missing:** Add the site's module initialization command to **Additional environment setup commands**.

**Container paths do not exist:** Check binds and the calculation PYTHONPATH from inside the container.

**No trajectory cell:** Supply a cell override in Angstrom and the appropriate PBC.
Check the original simulation metadata before assigning a cell.

**No TACAW chunks fit:** Reduce skip, chunk size, or chunk step.
The supplied engine excludes a chunk whose span ends exactly at the final frame boundary.

**ROI extends beyond propagated bandwidth:** Reduce reciprocal-space bounds or increase the full grid.
The adapter completes missing central-ROI crop metadata in the supplied engine and rejects cropped shapes that cannot be stored consistently.

**The scheduler state is UNKNOWN:** Check `squeue` availability and scheduler access.
The interface treats an unknown job as potentially active and prevents another submission.

**Preparation failed halfway:** Read the logs and generate a fresh run folder.
A converted input is published only when complete.
Initial execution never overwrites an existing result store.

**NumPy conversion needs too much memory:** Reduce the output chunk size in a future calculation by reducing its scan batch shape.
Conversion must decode each existing compressed chunk in full.
The generated NumPy file is stored on the calculation machine.

**The browser test skips:** Install Chromium through Playwright or set `TACAW_BROWSER_PATH`.
On HPC hosts where Chromium cannot reserve its virtual heap, install Firefox and set `TACAW_BROWSER=firefox`.
The scientific and Streamlit tests can run without a browser.
