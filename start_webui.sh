#! /bin/bash

function add_host_mapping() {
    if ! grep -q "webui" "/etc/hosts"; then
        echo "127.0.0.1    webui" | sudo tee -a "/etc/hosts"
        echo "Added webui to /etc/hosts"
    else
        echo "webui already exists in /etc/hosts"
    fi
}

function remove_host_mapping() {
    if grep -q "webui" "/etc/hosts"; then
        sudo sed -i "/webui/d" "/etc/hosts"
        echo "Removed webui from /etc/hosts"
    else
        echo "webui not found in /etc/hosts"
    fi
}

source /home/celiu/projects/codex_v2/codex/webui/.venv/bin/activate
add_host_mapping
trap remove_host_mapping EXIT

export PYTHONPATH=.

streamlit run webui/app.py --server.port=8501 --server.address=0.0.0.0 >> "/home/celiu/projects/codex_v2/codex/webui/webui.log" 2>&1