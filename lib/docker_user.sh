# Resolve the --user flag that makes container-written files land as
# $(id -u):$(id -g) on the host, regardless of rootless vs rootful Docker.
#
# - rootless dockerd: only container uid/gid 0 (root) maps back to the host
#   user invoking the daemon; any other uid falls into the subuid/subgid
#   remap range and lands on the host as a different, unrelated owner.
# - rootful dockerd: uids map 1:1, so use the real host uid/gid directly.
#
# Usage: source this file, then pass "${docker_user_flag[@]}" to `docker run`.
# Requires bash (uses arrays) — source it from a bash script, run with
# `bash script.sh` or `./script.sh` (with a `#!/usr/bin/env bash` shebang),
# not `sh script.sh` (dash has no arrays).
if docker info --format '{{json .SecurityOptions}}' 2>/dev/null | grep -q 'name=rootless'; then
    docker_user_flag=(--user 0:0)
else
    docker_user_flag=(--user "$(id -u):$(id -g)")
fi
