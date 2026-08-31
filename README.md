# LFVDDP

LFVDDP is a research codebase for learning a likelihood-ratio test statistic that
distinguishes statistically different pairs of physically equivalent collider
datasets. It supports generated studies and ROOT datasets, local single runs,
parallel training on the WIS ATLAS cluster, and plots that summarize convergence
and statistical performance.

## Workflow

1. Set up and activate the project environment.
2. Copy one of the tracked configuration packs and adapt it to the study.
3. Run one training locally, or submit an ensemble to the cluster.
4. Continue incomplete cluster submissions when needed.
5. Generate plots from the saved submission directory.

## Prerequisites

- Git and access to this repository.
- Python 3.11 or newer for the initial setup command.
- A mounted [CVMFS](https://cvmfs.readthedocs.io/en/stable/) installation. The
  project selects its pinned CVMFS Python environment automatically.
- For cluster submission: access to the WIS ATLAS PBS cluster, its `qsub`
  command, and Singularity. Connect through the institute network or VPN as
  required by WIS.

TensorFlow is installed only on non-macOS systems. This affects the optional
NPLM training path; the regular PyTorch workflow is available on macOS.

## Setup

Unless noted otherwise, run the commands below from the repository root.

Clone the repository and its submodules, then run the setup script from the
repository root:

```bash
git clone https://github.com/eladushiki/lfvddp.git
cd lfvddp
git submodule update --init --recursive
source scripts/setup_python_environment.sh
```

The script creates `.venv`, installs UV into it, and synchronizes the exact
dependencies in `uv.lock`. To use the existing environment in a later shell,
run this from any directory:

```bash
source /path/to/lfvddp/scripts/activate_python_environment.sh
```

Do not create a second environment or install a separate requirements file. To
use a different UV download cache, export `UV_CACHE_DIR` before running the
setup script.

## Configuration

Every run is configured with JSON or YAML files. Start from the pack that
matches the data source:

- [`configs/basic-generated`](configs/basic-generated) generates all datasets.
- [`configs/basic-loaded`](configs/basic-loaded) loads ROOT data and can inject
  a generated signal.

Copy a pack before editing it:

```bash
cp -R configs/basic-generated configs/my-study
```

Local configuration packs below `configs/` are ignored by Git. Each tracked
pack contains separate files for the dataset, detector, training, plotting,
cluster, and user settings.

Pass files or directories after `--configs`. Directories are searched
recursively for JSON and YAML files. Files are shallow-merged in the order they
are resolved, and later values override earlier ones. This makes a small
override file useful when most settings should remain unchanged:

```bash
python train/single_train.py \
  --configs configs/basic-generated configs/my-overrides.json \
  --debug
```

Important configuration choices include:

- `config__runtag` names the study and `config__out_dir` selects its output root.
- `random_seed` reproduces a fresh run when supplied; otherwise LFVDDP creates
  and records one.
- `dataset__definitions` describes the A/B signal-region and control-region
  samples. See the two basic packs for generated and loaded examples.
- `train__epochs`, checkpoint frequency, network width, and learning-rate
  settings control optimization.
- The default nuisance model is binned and requires bin minima, maxima, and
  counts. For a neural nuisance model, set
  `train__nuisance_is_neural_network` to `true`, remove the bin settings, and
  provide `train__nuisance_nn_inner_layer_nodes`.
- `plot__plot_specifications` selects the plots produced for a submission. Plot
  behavior is documented in [`plot/specs`](plot/specs).
- `cluster__qsub_n_jobs`, resource requests, and walltime control cluster jobs.

Configuration is validated before work begins, so incompatible or incomplete
settings fail early with an explanatory error.

## Run locally

Activate the environment and run one training process:

```bash
python train/single_train.py --configs configs/my-study
```

Normal runs require a clean Git working tree so their saved commit identifies
the executed code. During development, `--debug` disables that check:

```bash
python train/single_train.py --configs configs/my-study --debug
```

Use `--out-dir <directory>` to override `config__out_dir` for a fresh run.

## Submit to the WIS ATLAS cluster

Run the submission command from a configured cluster login environment:

```bash
python train/submit_train.py --configs configs/my-study
```

The command stages the resolved configuration, submits the configured training
array, and then submits plotting after training completes. It reuses the
existing Singularity image by default.

Useful fresh-submission options are:

- `--build-container` builds a new image before training.
- `--only-train` skips both container building and automatic plotting.
- `--out-dir <directory>` overrides the configured output root.
- `--debug` permits submission from a dirty working tree.

The cluster user configuration supplies the repository URL, Singularity
executable and activation command. Set `cluster__uv_cache_dir` in the local pack
if cluster jobs need a persistent UV cache.

## Continue a cluster submission

Pass either a run directory or its `context.json`. The saved configuration,
seed, and checkpoints are restored automatically:

```bash
python train/submit_train.py --continue <run-directory-or-context.json>
```

If the saved epoch target was reached before convergence, replace it with a
larger target:

```bash
python train/submit_train.py \
  --continue <run-directory-or-context.json> \
  --epochs-target 750000
```

If the walltime budget was insufficient, extend it:

```bash
python train/submit_train.py \
  --continue <run-directory-or-context.json> \
  --extra-time 24:00:00
```

`--extra-time` uses `HH:MM:SS` and adds to the saved total. Long totals are split
according to `cluster__qsub_walltime_limit`; repeat the continuation command if
another scheduler chunk is required. A continuation may use only `--debug`,
`--epochs-target`, and `--extra-time` in addition to `--continue`.

## Create plots

Create the configured plots from one completed submission:

```bash
python plot/create_plots.py <submission-directory>
```

The submission must contain the staged `configs` directory and its training
runs. The plotting command reads that saved configuration; it does not accept
`--configs`. Use `--debug` while working with an uncommitted tree.

For configured plots that aggregate recursively across background and signal
submissions, use:

```bash
python plot/create_plots.py <multi-run-directory> --multi-run-plots
```

## Outputs and reproducibility

Runs are written below `config__out_dir` in a unique directory containing
`context.json`. The context records the resolved configuration paths, command,
Git commit, seed, submission history, completion state, and produced files.
Training histories, checkpoints, model weights, worker output, results, and
plots are stored alongside that context as applicable.

The `results/`, `data/`, and local configuration directories are intentionally
ignored by Git. Back up important run directories and external datasets
separately.

## Tests

After activating the project environment, run:

```bash
python -m pytest
```

## Repository map

- [`configs`](configs) contains the tracked example configuration packs and
  configuration validation.
- [`data_tools`](data_tools) defines datasets, generators, detectors, and
  statistical calculations.
- [`neural_networks`](neural_networks) contains LFVDDP and NPLM model code.
- [`train`](train) contains the local and cluster training entry points.
- [`plot`](plot) contains plot generation and the user-visible plot contracts.
- [`frame`](frame) contains execution, configuration, cluster, and file-handling
  infrastructure.
- [`paper_scripts`](paper_scripts) contains notebooks and scripts used for paper
  figures.

## Troubleshooting

- If setup cannot find the pinned Python setup file, verify that `/cvmfs` is
  mounted and accessible.
- If a loaded dataset uses a `root://` URL, verify network access to its XRootD
  endpoint. The environment already includes the XRootD backend for `fsspec`.
- If cluster submission cannot reach PBS or CVMFS, reconnect through the WIS
  network or VPN and confirm access on the cluster login node.
- If a non-debug run reports a dirty working tree, commit the intended code and
  configuration changes or use `--debug` only for exploratory work.
