#!/bin/bash

# SERVICES=$(find . -mindepth 1 -maxdepth 1 -type d)
SERVICES=("webui" "agent_service" "index_service")

for SERVICE in "${SERVICES[@]}"; do
  if [ -d "$SERVICE" ]; then
    echo "Processing $SERVICE..."
    
    if [ -f "$SERVICE/pyproject.toml" ]; then
      cd "$SERVICE" || exit
      
      if ! command -v poetry &> /dev/null; then
        echo "Error: Poetry is not installed. Please install Poetry and try again."
        exit 1
      fi
      
      source .venv/bin/activate
      poetry env use python
      poetry lock
      poetry export -f requirements.txt --without-hashes -o requirements.txt
      deactivate
      echo "requirements.txt generated for $SERVICE"
      
      cd - > /dev/null
    else
      echo "Warning: $SERVICE does not contain pyproject.toml. Skipping..."
    fi
  else
    echo "Warning: $SERVICE directory not found. Skipping..."
  fi
done

echo "All services processed."
