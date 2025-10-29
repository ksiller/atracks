#!/usr/bin/env bash

set -e

if [[ -z "$*" ]]; then
    exec bash --login
else
    exec "$@"
fi

