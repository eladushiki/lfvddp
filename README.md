# LFVNN Symmetrized - Overview

> The LFVNN Symmetrized project aims to harness the power of the machine to look for deviations from the Standard Model in existic high energy physics collider data.

This project runs on the WIS's ATLAS cluster, as well as on your computer, to use our machinery on real or simulated physical datasets.

# Contents
- `configs` directory contain configuraion files necessary to run the project. An example or template for each is included and meant to be copied by the user for modification.
- `data_tools` directory handles mathematical and statistical calculations needed to operate this project.
- `frame` is a place for all framework tools needed for this project to run and communicate.
- `mattiasdata` has some useful (legacy) datasets (and frankly, needs to be cleaned)
- `neural_networks` handles NN tools operated here, as well as `NPLM` tools.
- `paper_scripts` has specific code to reproduce figures used in each of our papers. This is an interactive section which is mostly made of `jupyter` notebooks.
- `plot` defines the tools to create plots.
- `train` handles physical datasets and weighs hypotheses for new physics, returning a profile likelihood goodness of fit measure.
- Additional project files such as `.gitignore` and this very file.

# Installation Steps
## Setup
Start by cloning this repository to your desired location, using

> git clone \<url for clone\> [\<desired direcotory name\>]

> git submodule update --init --recursive

You need an operative Python interpreter to run this project. You can choose any of the following options

### CERN VM FS (Recommended):

`source` the a Python interpreter each time you open a shell (may be configured to happen automatically in an IDE) from CERN's CVMFS, using:

> source  /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc12-opt/setup.sh

installation instructions for the filesystem can be found in [this link](https://cvmfs.readthedocs.io/en/stable/cpt-quickstart.html).

WSL: I found it also necessary to use

> sudo apt-get install libicu-dev

upon installation, and

> echo <your password> | sudo -S cvmfs_config wsl2_start

Each time the systems is up (may be configured to happen automatically in vairous ways)
### Local Venv:

Use your local python installation, creating a virtual environment for the installation of the specific dependency version needed. Any newer Python interpreters should work, but run up untill now are done with Python 3.9 to 3.10.

Then, it is customary to run

> pip install -e .

to install the project to run locally with dependecies and being able to run the code locally while editing it in place.

## Environment Configuration

### VPN

Call 4444 for WIS IT or read guide for help with insatlling WIS VPN.

The communication with the cluster and CVMFS is enables from within the WIS (or equivalent) network.

### ATLAS Cluster SSH connection

There is adifferent way to connect to WIS ATLAS cluster, whether you're inside WIS_Secure wifi or outside the institute using a VPN.

Starting form the simple part - if you're in the institute you should be able to SSH seamlessly to the cluster using your username and password, after creating one - Contact the department's lcg-managers@weizmann.ac.il for help on this (or any cluster related issue).

There is a more complex procedure if you're using a VPN to connect from outside the institute which requires you to use a proxy jump (as of the time of writing this). You can contact Elad to save time on configuring this, or follow the internal guide.

## Job Submission to WIS ATLAS Cluster [under construction]

> NOTE: Submitting from your own PC is under construction. Run anything from an SSH connection for now.

Generally, to also be able to run on the cluster, the same command sould be also run there. The following command might be of use beforehand:

> export PATH="$PATH:/srv01/agrp/<your-username>/.local/bin"

if pip is not already recognized.

To submit SGE jobs to the WIS cluster, you should install slurm. The [specific isntallation command](https://command-not-found.com/qsub) depends on your OS.

## IDE configuration (examle in VSCode)

Use dialog (`ctrl+shift+P`) to create or create manualy a `launch.json` file. In them, configure basic running configuration for the project to run easily. For exmaple,

```json
{
    "version": "0.2.0",
    "configurations": [
        { // DEBUG Single train
            "name": "[DEBUG] Single train with config files",
            "type": "debugpy",
            "request": "launch",
            "program": "train/single_train.py",
            "console": "integratedTerminal",
            "args": [
                "--user-config", "${config:myConfig.userConfig}",
                "--cluster-config", "${config:myConfig.clusterConfig}",
                "--dataset-config", "${config:myConfig.datasetConfig}",
                "--train-config", "${config:myConfig.trainConfig}",
                "--plot-config", "${config:myConfig.plotConfig}",
                "--debug",
            ]
        },
        { // Submit train
            "name": "Submit train with config files",
            "type": "debugpy",
            "request": "launch",
            "program": "train/submit_train.py",
            "console": "integratedTerminal",
            "args": [
                "--user-config", "${config:myConfig.userConfig}",
                "--cluster-config", "${config:myConfig.clusterConfig}",
                "--dataset-config", "${config:myConfig.datasetConfig}",
                "--train-config", "${config:myConfig.trainConfig}",
                "--plot-config", "${config:myConfig.plotConfig}",
            ]
        },
        { // DEBUG plot
        "name": "[DEBUG] plot",
        "type": "debugpy",
        "request": "launch",
        "program": "plot/create_plots.py",
        "console": "integratedTerminal",
        "args": [
            "--user-config", "${config:myConfig.userConfig}",
            "--cluster-config", "${config:myConfig.clusterConfig}",
            "--dataset-config", "${config:myConfig.datasetConfig}",
            "--train-config", "${config:myConfig.trainConfig}",
            "--plot-config", "${config:myConfig.plotConfig}",
            "--debug",
        ]
    }
}
```

The custom file paths here refer to a different file, `settings.json`, of the form:
```json
{
    "myConfig.userConfig": "configs/user/<name>.json",
    "myConfig.clusterConfig": "configs/cluster/<your config>.json",
    "myConfig.datasetConfig": "configs/dataset/<your config>.json",
    "myConfig.trainConfig": "configs/train/<your config>.json",
    "myConfig.plotConfig": "configs/plot/<your config>.json",
}
```
Which you create and direct to.

To configure a custom terminal `source`ing the environmet as explained above, you can create a custom rc file (text file) and write said commands in it. i.e., for WSL2:

```bash
source  /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc12-opt/setup.sh
set TF_USE_LEGACY_KERAS=True  # Set legacy Keras usage, needed for NPLM
```

To have it run automatically when running stuff, have the "integrated terminal" (seen in above `launch.json`) configured by having:
```json
{
    // ...existing settings...
    "terminal.integrated.defaultProfile.windows": "WSL with Custom Commands",
    "terminal.integrated.profiles.windows": {
        // ...existing profiles...
        "WSL with Custom Commands": {
            "path": "C:\\Windows\\System32\\wsl.exe",
            "args": ["--exec", "bash", "-c", "source <your rc path> && echo 'Custom WSL Terminal Initialized' && exec bash"],
            "icon": "terminal-bash"
        }
    }
}
```
and choose with VSCode command "Terminal: Select Default Profile" the "WSL with Custom Commands".

# Design Concepts Details
## Everyting runs in a context

To be able to keep track of every run and its products, a running context is implemented. It does a few things:

- Records the current running configuration
- Records the current Git commit
- To do so reliably, forces you to commit changes before running
- Enables the user to add any additional context as new parameters of the context to be documented
- Documents time of run
- Forces seeding of numpy random, for reproducibility of the results
- Saves it in a file, that should be adjacent to the resulting output

What is not version controlled:

- External databases used, should be version controlled separately (although their locations are).
- That inclueds the version of NN's and such (although their locations are).
- Virtual environment dependency versions are not (yet) well documented.
- Any configuration of external tools. Any scripts in user defined "scripts_dir".

To use this functionality in any new entry point, run the main function inside the context.

**This means that we are not allowed to tamper with git history. Amending or rewriting commits in any other way is prohibited, for it may create results that are non-reproducible!**

## Configuration files
The configuration files and then the Config* dataclasses are the structures that should contain all the parameters that are run-individual.

While Config* classes' contents are divided logically to different classes, the program needs to be called with each of the `.json` configuration file types, as long as they contain togehter all the necessary parameters. This is implemented so to enable single-file-for-run usage, as well as separation for personal privacy and context-dependent needs [Under construction].

Is is specifically recommended that personal username and password for SSH connection with the WIS cluster would be stored in a separate file and not added to git. This is why the example `basic_user_config.json` file contains `cluster__*` parameters, which later end up in the `ClusterConfig` dataclass.

## Plotting

Any function that is implemented in `plot/plots.py` can be called by name from the "name" field in a `plot_config.json` file. It is called with keyword arguments as specified in the `instructions` field inside (see `basic_plot_config.json` for example).

To implement any new plot, simply define its generating function there in the form of:
```python
def plot_something_new(context: ExecutionContext, **kwargs) -> matplotlib.figure.Figure:
	...
```
and you should be able to use it right away through `create_plots.py`.

# Entry Points
## Training

Training entry points:
- `single_train.py` for the server to run each time (and local tests)
- `submit_train.py` for remote submission of multiple copies of `single_train.py` [Currently only in-place, when running at the ATLAS cluster]

## Plotting
- `create_plots.py` would follow the instruction in the configuration files to gather the necessary data from completed trainings and producte the plots
