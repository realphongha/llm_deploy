#!/bin/bash

HOST=""
SSH_PORT=22
API_PORT=8008
HOST_SET=0
SSH_PORT_SET=0
API_PORT_SET=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Forward a local API port to a remote host over an SSH reverse tunnel.

Options:
  -h, --help             Show this help and exit
  -H, --host HOST        Target host (required)
  -p, --ssh-port PORT    SSH port (default: 22)
  -a, --api-port PORT    API port to forward (default: 8008)

Example:
  $(basename "$0") --host {target_server_ip} --ssh-port 22 --api-port 8008
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)     usage; exit 0 ;;
    -H|--host)     HOST="$2"; HOST_SET=1; shift 2 ;;
    -p|--ssh-port) SSH_PORT="$2"; SSH_PORT_SET=1; shift 2 ;;
    -a|--api-port) API_PORT="$2"; API_PORT_SET=1; shift 2 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

# Prompt for any argument that was not passed; press Enter to keep the default.
prompt_var() {
  local -n ref="$1"
  local prompt="$2"
  read -rp "$prompt [$ref]: " input
  [[ -n "$input" ]] && ref="$input"
}

if [[ $HOST_SET -eq 0 ]]; then
  read -rp "Target host: " input
  HOST="$input"
fi
if [[ -z "$HOST" ]]; then
  echo "Error: target host is required." >&2
  usage >&2
  exit 1
fi

[[ $SSH_PORT_SET -eq 0 ]] && prompt_var SSH_PORT "SSH port"
[[ $API_PORT_SET -eq 0 ]] && prompt_var API_PORT "API port to forward"

_sigint() {
  echo "Received Ctrl‑C, terminating." >&2
  trap - SIGINT    # reset SIGINT to default
  kill -SIGINT "$$"
}

trap _sigint SIGINT

while true; do
  ssh -p "$SSH_PORT" -R "$API_PORT:localhost:$API_PORT" "$HOST" -N
  rc=$?
  echo "Exited with status $rc, restarting…" >&2
  sleep 1
done
