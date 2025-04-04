#! /bin/bash

function add_host_mapping() {
    if ! grep -q "index_service" "/etc/hosts"; then
        echo "127.0.0.1    index_service" | sudo tee -a "/etc/hosts"
        echo "Added index_service to /etc/hosts"
    else
        echo "index_service already exists in /etc/hosts"
    fi
}

function remove_host_mapping() {
    if grep -q "index_service" "/etc/hosts"; then
        sudo sed -i "/index_service/d" "/etc/hosts"
        echo "Removed index_service from /etc/hosts"
    else
        echo "index_service not found in /etc/hosts"
    fi
}

source /home/celiu/projects/codex_v2/codex/index_service/.venv/bin/activate
add_host_mapping
trap remove_host_mapping EXIT
uvicorn index_service.app:app --host 0.0.0.0 --port 8102 --reload >> "/home/celiu/projects/codex_v2/codex/index_service/index_service.log" 2>&1