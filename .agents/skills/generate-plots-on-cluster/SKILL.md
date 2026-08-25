---
name: generate-plots-on-cluster
description: "Run many training in parallel using the WIS ATLAS cluster infrastructure via ssh"
---

# Generate Plots on Cluster

## Purpose

<!-- Describe the outcome this skill should achieve. -->
Review jobs that finished running recently, and run the post process for generating descriptive plots by their spec.

## When to Use

<!-- Describe the request types, conditions, or phrases that should trigger this skill. -->
When requested to review done jobs, generate recent plots or something of sort.

## Definitions

### Commands

- SSH Command: `ssh <WIS_CLUSTER_PLOT_SSH_TARGET>`
- Arrayed Qstat Command: `qstat -tu $USER`
- Finished Qstat Command: `qstat -xu $USER`

### Paths

- Submission State File Path: `.agents/submission-state.yaml`
- Project Root at Remote: `<WIS_CLUSTER_REMOTE_PROJECT_ROOT>`
- Relative Launch File Path: `.vscode/launch.json`
- Relative Results Directory: `results/`
- Relative Highlights Directory: `results/highlights`
- Relative Packs Directory: `configs/packs`
- Relative Plots Directory: `results/plots/`

## Local configuration

Before using this skill, add these machine-specific values to the untracked `.gsd/SECRETS.md` file:

- `WIS_CLUSTER_PLOT_SSH_TARGET`: SSH target in the form `<username>@<host>`.
- `WIS_CLUSTER_REMOTE_PROJECT_ROOT`: Absolute path to this repository on the cluster.
- `WIS_CLUSTER_SSH_IDENTITY_FILE` (optional): Absolute path to a non-default private-key file.

Configure the key with `~/.ssh/config` or the SSH agent. Never commit a username, host, remote path, private-key path, or private key to this skill or the repository.

## Prerequisites

<!-- List required access, environment, configuration, data, and branch state. -->
- After verifying that the WIS VPN is connected.
- After opening an ssh connection with SSH Command.
- After checking the running jobs using Arrayed Qstat Command over ssh and comparing to the saved job ids from last submission.
- After checking for jobs that finished running entirely with Finished Qstat Command.

## Inputs

<!-- List the inputs and their expected formats. -->
Description of which jobs were ran and not yet reviewed and plotted from Submission State File Path.

## Procedure

<!-- Write the safe, ordered steps for preparing and submitting work to the cluster. -->
- Ask the user or verify that the WIS VPN is connected.
- Start an ssh session.
- `cd` to the root directory of the project, Project Root at Remote.
- Read remote Relative Launch File Path to get an idea of the current plotting syntax. Replace path input arguments with the relative paths.
- Use "[DEBUG] plot with prompt for target" configuration for each finished array job, with the target it's parent submission directory in Relative Results Directory. Dir name should contain `...run_of_submit.py...`.
- Organize plotted runs in Relative Highlights Directory, in a manner that resembles the config packs the plots were generated from in Relative Packs Directory. Structure should not be identical, but each significance plot should be possible to generate from withing a single parent directory.
- Generate significance progression plots from parent directories, with a command that mimics "[DEBUG] plot with prompt for target" from Relative Launch File Path.
- Wait for plots to finish being produced.
- Terminate the ssh session and return to the local environment. `scp` the plots back to local Relative Plots Directory.
- Update processing state of submitted jobs in yaml file.

## Verification

<!-- State how to confirm that submission succeeded and where to inspect logs/results. -->
Each run should generate plots according to target submission output directory's `configs/plot_config.json`.

## Failure Handling

<!-- Record common failure modes, diagnosis steps, and recovery actions. Do not include secrets. -->
If encountered any problems, ask me what to do.

## Output and Handoff

<!-- State what artifacts, job identifiers, result paths, and next actions must be recorded. -->
Upon successful submission, plots should be found in local dir and the state of submission analysis should be updated.

## Safety Constraints

<!-- State any required confirmation gates and operations that must not be automated. -->
You're not allowed to run any commands on the cluster unless specifically instructed to do so by me.
