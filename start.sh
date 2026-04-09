#!/bin/bash
# Start the Model Runner server
cd "$(dirname "$0")/backend"
exec python main.py "$@"
