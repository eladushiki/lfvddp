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
bootstrap_command='set +u; export COMPILER="${COMPILER:-gcc}"; export CXX="${CXX:-}"; export MANPATH="${MANPATH:-}"; source scripts/activate_python_environment.sh'

if [ "$#" -eq 0 ]; then
    interactive_command="$bootstrap_command; exec /bin/bash --noprofile --norc -i"
    quoted_interactive_command=$(shell_quote "$interactive_command")
    remote_command="cd $quoted_root && exec /bin/bash -lic $quoted_interactive_command"
    if [ -n "$identity_file" ]; then
        exec ssh -tt -i "$identity_file" "$ssh_target" "$remote_command"
    fi
    exec ssh -tt "$ssh_target" "$remote_command"
fi

command_text=$*
run_command="$bootstrap_command; $command_text"
quoted_run_command=$(shell_quote "$run_command")
remote_command="cd $quoted_root && exec /bin/bash -lc $quoted_run_command"

if [ -n "$identity_file" ]; then
    exec ssh -i "$identity_file" "$ssh_target" "$remote_command"
fi
exec ssh "$ssh_target" "$remote_command"
