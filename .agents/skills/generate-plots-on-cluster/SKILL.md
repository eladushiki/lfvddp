---
name: generate-plots-on-cluster
description: "Run many training in parallel using the WIS ATLAS cluster infrastructure via ssh"
---

# Submit on Cluster

## Purpose

<!-- Describe the outcome this skill should achieve. -->
Submit a job on the ATLAS cluster at WIS using eladklig's credentials.

## When to Use

<!-- Describe the request types, conditions, or phrases that should trigger this skill. -->
When requested to review done jobs, generate recent plots or something of sort.

## Prerequisites

<!-- List required access, environment, configuration, data, and branch state. -->
- After verifying that the WIS VPN is connected.
- After opening an ssh connection with `ssh eladklig@wipp-external`.
- After checking the running jobs using `qstat -tu $USER` over ssh and comparing to the saved job ids from last submission.
- After checking for jobs that finished running entirely with `qstat -xu $USER`.

## Inputs

<!-- List the inputs and their expected formats. -->
Description of which jobs were ran and not yet reviewed and plotted from `.agents/skills/submit-on-cluster/state.yaml`

## Procedure

<!-- Write the safe, ordered steps for preparing and submitting work to the cluster. -->
- Ask the user or verify that the WIS VPN is connected.
- Start an ssh session.
- `cd` to the root directory of the project, `/storage/agrp/eladklig/SymmetrizedDDP`.
- Read remote `.vscode/launch.json` to get an idea of the current plotting syntax. Replace path input arguments with the relative paths.
- Use "[DEBUG] plot with prompt for target" configuration for each finished array job, with the target it's parent submission directory in `results/`. Dir name should contain `...run_of_submit.py...`.
- Organize plotted runs in `results/highlights/` directory, in a manner that resembles the config packs the plots were generated from in `configs/packs`. Structure should not be identical, but each significance plot should be possible to generate from withing a single parent directory.
- Generate significance progression plots from parent directories, with a command that mimics "[DEBUG] plot with prompt for target" from `.vscode/launch.json`.
- Wait for plots to finish being produced.
- Terminate the ssh session and return to the local environment. `scp` the plots back to local `results/plots/`.
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
