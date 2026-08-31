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

1. Run `scripts/ssh-to-cluster.sh` from the local repository root in a
   persistent terminal session. The helper opens SSH and starts a login shell
   in `WIS_CLUSTER_REMOTE_PROJECT_ROOT`.
2. Reuse that terminal session for every cluster command in the workflow.
3. Verify the connection with `pwd` and `git status --short --branch` before
   doing work.
4. Exit the connection only after plotting, submission, and verification are
   complete.

For a bounded non-interactive check, pass one shell command string to the
helper. It runs from the same remote project root and login environment:

```sh
scripts/ssh-to-cluster.sh 'pwd && git status --short --branch'
```

## Failure handling

Retry connection failures up to three times. If the host remains unreachable,
ask the user to connect the WIS VPN and complete its 2FA; only the user can do
that. Do not attempt cluster work through a different host or connection.

## Safety

Opening the connection does not authorize arbitrary cluster mutations. The
calling task or downstream skill must provide that authorization.
