---
name: submit-on-cluster
description: "Generate plots for done jobs"
---

# Submit on Cluster

## Purpose

<!-- Describe the outcome this skill should achieve. -->
Review jobs that finished running recently, and run the post process for generating descriptive plots by their spec.

## When to Use

<!-- Describe the request types, conditions, or phrases that should trigger this skill. -->
When requested to "run jobs" or "submit jobs" on "the cluster", "WIS cluster", "ATLAS cluster".

## Definitions
### Bash Commands
- Active Job Count Command: `qstat -tu $USER | wc -l`
- Queued Job Count Command: `qstat -tu $USER | grep Q | wc -l`
- User SSH Command: `ssh eladklig@wipp1-external`
- Run Verification Command: `qstat -wu $USER`

### File and Dir Paths
#### Remote
- Packs Parent Directory: `/storage/agrp/eladklig/SymmetrizedDDP/configs/packs/`
- Remote Project Root `/storage/agrp/eladklig/SymmetrizedDDP`
#### Relative
- Launch Path `.vscode/launch.json`
- Submitted Job State File: `.agents/skills/submit-on-cluster/state.yaml`

### Etc.
- Submit Train Launch Option: "[DEBUG] Submit train with prompt"
- Runtag Config Field: `config__runtag`

## Prerequisites

<!-- List required access, environment, configuration, data, and branch state. -->
- After creating or locating a configs package to use in the submission.
- After verifying that the WIS VPN is connected.
- After checking the amount of running jobs using Active Job Count Command over ssh, and the number of queued jobs using Queued Job Count Command to ensure the submisison will not pass the quota of 1000 queued jobs. The number of allowed running jobs depend on the resources requested.
- Record the commit hash of the local `HEAD` and branch. Ask user which branch or commit to use for the submission.

## Inputs

<!-- List the inputs and their expected formats. -->
A specified directory that contains config files with all required attributes for a submission.

## Procedure

<!-- Write the safe, ordered steps for preparing and submitting work to the cluster. -->
- Ask the user or verify that the WIS VPN is connected.
- `scp` the config pack to the same adequate location under Packs Parent Directory from local one, after reading its contents.
- Start an ssh session with User SSH Command.
- `cd` to the root directory of the project, Remote Project Root.
- Look at remote git status. If on requested branch and no local changes, you may continue. Otherwise, stop and ask the user what to do.
- Read remote Launch Path to get an idea of the updated submission command. Replace path input arguments with the relative paths. Copy Submit Train Launch Option configuration.
- Check that the job is created using `qstat` commands.
- Terminate the ssh session and return to the local environment.
- Document ran job ids and their purpose, each in a different line, in Submitted Job State File. Save their state - submitted and not yet analyzed. Document the plot name and the config pack from which they were generated (from Packs Parent Directory).

## Verification

<!-- State how to confirm that submission succeeded and where to inspect logs/results. -->
Use Run Verification Command and look for the job with a name that matches `train_config.json`'s Runtag Config Field field.

## Failure Handling

<!-- Record common failure modes, diagnosis steps, and recovery actions. Do not include secrets. -->
If submission is rejected due to exceeding the quota of queued jobs, forward the error to the user and stop.

## Output and Handoff

<!-- State what artifacts, job identifiers, result paths, and next actions must be recorded. -->
Upon successful submission, no output is required, except for an indication that the job was submitted successfully. The added jobs should be documented.

## Safety Constraints

<!-- State any required confirmation gates and operations that must not be automated. -->
You're not allowed to run any commands on the cluster unless specifically instructed to do so by me.
