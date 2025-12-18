#!/bin/bash

# Helper script to sync uv environment with GPU or CPU configuration
# Usage: ./sync_env.sh [gpu|cpu]

sync_env() {
    local env_type=$1

    if [[ "$env_type" != "gpu" && "$env_type" != "cpu" ]]; then
        echo "Error: Invalid argument. Use 'gpu' or 'cpu'"
        echo "Usage: ./sync_env.sh [gpu|cpu]"
        return 1
    fi

    local source_file="pyproject_${env_type}.toml"

    if [[ ! -f "$source_file" ]]; then
        echo "Error: $source_file not found"
        return 1
    fi

    echo "Copying $source_file to pyproject.toml..."
    cp "$source_file" pyproject.toml

    echo "Running uv sync..."
    uv sync

    local sync_status=$?

    echo "Removing pyproject.toml..."
    rm pyproject.toml

    if [[ $sync_status -eq 0 ]]; then
        echo "Successfully synced $env_type environment"
    else
        echo "Error: uv sync failed"
        return $sync_status
    fi
}

# Call the function with the first argument
sync_env "$1"
