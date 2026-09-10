#!/usr/bin/env bash
# Recursively chown a file/dir back to your own host user, by running a
# throwaway root container over it. Needed because containers (esp. the
# vibe/vLLM ones that write as root, or ones run under a different --user)
# can leave files owned by uids that don't belong to you on the host.
#
# Usage: ./claim_back_permission.sh <path> [image]
#
# Why not just `sudo chown -R "$(id -u):$(id -g)" <path>`? Because these
# hosts run everything through Docker without host sudo access assumed;
# a root container achieves the same thing via the container runtime.
#
# See lib/docker_user.sh for the full rootless/rootful uid-mapping story.
# The short version: under rootless Docker, only container uid/gid 0 (root)
# maps back to the host user that started the daemon (you); any other
# container uid lands on some unrelated subordinate uid. Under rootful
# Docker there's no remapping, so container root really is host root, and
# you must chown to your real host uid/gid instead.
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <path> [image]" >&2
    exit 1
fi

target=$(readlink -f "$1")
image=${2:-llama-cpp}

if [ ! -e "$target" ]; then
    echo "error: path does not exist: $target" >&2
    exit 1
fi

if docker info --format '{{json .SecurityOptions}}' 2>/dev/null | grep -q 'name=rootless'; then
    # rootless: container uid/gid 0 maps back to your host user
    chown_id="0:0"
else
    # rootful: uids map 1:1, chown to your real host uid/gid
    chown_id="$(id -u):$(id -g)"
fi

echo "chowning $target to $(id -un):$(id -gn) via container uid $chown_id ($image)"

docker run --rm \
    -v "$target":/claim_back_permission_target \
    --entrypoint "" \
    "$image" \
    chown -R "$chown_id" /claim_back_permission_target

echo "done: $(ls -lnd "$target")"
