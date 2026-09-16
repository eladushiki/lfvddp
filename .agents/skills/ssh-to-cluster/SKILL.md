---
name: ssh-to-cluster
description: "Open the shared SSH session used for work in the WIS ATLAS project checkout."
---

# SSH to Cluster

Open one reusable SSH session at the remote project root. Cluster workflow
skills assume this connection already exists; they must not define or open
their own SSH connections.

## Local configuration

Read these untracked values from `.gsd/SECRETS.md` without printing them:

- `WIS_CLUSTER_SSH_TARGET`: SSH target in `<username>@<host>` form.
- `WIS_CLUSTER_REMOTE_PROJECT_ROOT`: absolute remote checkout path.
- `WIS_CLUSTER_SSH_IDENTITY_FILE` (optional): non-default identity file.

Never commit connection values or credentials.

## Procedure

1. Run `.agents/skills/ssh-to-cluster/scripts/ssh-to-cluster.sh` from the local repository root in a
   persistent terminal session. The helper opens SSH at
   `WIS_CLUSTER_REMOTE_PROJECT_ROOT` and starts a clean Bash shell with the
   project venv activated.
   When launched by Codex, request the elevated network permission: the
   restricted shell cannot resolve the cluster host.
2. Verify that `python -c 'import torch'` succeeds. The helper starts clean
   Bash and supplies default `COMPILER`, `CXX`, and `MANPATH` values before the
   Bash-specific project activation. The generated CVMFS scripts dereference
   those variables under `nounset`; a plain remote zsh login leaves them unset.
   Do not replace this bootstrap with `/usr/bin/python`.
3. Reuse that terminal session for every cluster command in the workflow.
4. Verify the connection with `pwd` and `git status --short --branch` before
   doing work.
4. In a newly created cluster checkout, initialize its own locked environment
   once in clean Bash with `export COMPILER=gcc CXX='' MANPATH=''; source
   scripts/setup_python_environment.sh`. Later shells use the helper's normal
   activation. Do not borrow or bind another checkout's `.venv`.
5. Exit the connection only after plotting, submission, and verification are
   complete.

For a bounded non-interactive check, pass one shell command string to the
helper. It runs from the same remote project root and login environment:

```sh
.agents/skills/ssh-to-cluster/scripts/ssh-to-cluster.sh 'pwd && git status --short --branch'
```

## Failure handling

Make one connection attempt. If it fails because the network, DNS, host, VPN,
or interactive 2FA is unavailable, stop immediately, leave workflow state
unchanged, and ask the user to restore access. Wait for the user before trying
again; do not retry automatically or use a different host or connection.

## Safety

Opening the connection does not authorize arbitrary cluster mutations. The
calling task or downstream skill must provide that authorization.
