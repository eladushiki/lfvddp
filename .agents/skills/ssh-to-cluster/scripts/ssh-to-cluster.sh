#!/bin/sh
set -eu

secrets_file=.gsd/SECRETS.md

read_secret() {
    awk -F= -v key="$1" '$1 == key {print substr($0, index($0, "=") + 1); exit}' "$secrets_file"
}

shell_quote() {
    escaped=$(printf '%s' "$1" | sed "s/'/'\\\\''/g")
    printf "'%s'" "$escaped"
}

ssh_target=$(read_secret WIS_CLUSTER_SSH_TARGET)
remote_project_root=$(read_secret WIS_CLUSTER_REMOTE_PROJECT_ROOT)
identity_file=$(read_secret WIS_CLUSTER_SSH_IDENTITY_FILE)

if [ -z "$ssh_target" ]; then
    printf '%s\n' 'WIS_CLUSTER_SSH_TARGET is not configured in .gsd/SECRETS.md' >&2
    exit 1
fi
if [ -z "$remote_project_root" ]; then
    printf '%s\n' 'WIS_CLUSTER_REMOTE_PROJECT_ROOT is not configured in .gsd/SECRETS.md' >&2
    exit 1
fi

quoted_root=$(shell_quote "$remote_project_root")

if [ "$#" -eq 0 ]; then
    remote_command="cd $quoted_root && exec \${SHELL:-/bin/sh} -l"
    if [ -n "$identity_file" ]; then
        exec ssh -tt -i "$identity_file" "$ssh_target" "$remote_command"
    fi
    exec ssh -tt "$ssh_target" "$remote_command"
fi

command_text=$*
quoted_command=$(shell_quote "$command_text")
remote_command="cd $quoted_root && exec \${SHELL:-/bin/sh} -lc $quoted_command"

if [ -n "$identity_file" ]; then
    exec ssh -i "$identity_file" "$ssh_target" "$remote_command"
fi
exec ssh "$ssh_target" "$remote_command"
