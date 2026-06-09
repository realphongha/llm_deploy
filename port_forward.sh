#!/bin/bash

_sigint() {
  echo "Received Ctrl‑C, terminating." >&2
  trap - SIGINT    # reset SIGINT to default
  kill -SIGINT "$$"
}

trap _sigint SIGINT

while true; do
  ssh -R 8008:localhost:8008 rack -N
  rc=$?
  echo "Exited with status $rc, restarting…" >&2
  sleep 1
done
