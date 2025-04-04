import json

import requests

from shared import constants
from webui import app_config


def send_message(message, history, model="llama3.1") -> str:
    url = "http://{app_config.local_llm_url}/api/chat"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": history + [{"role": "user", "content": message}],
    }

    response = requests.post(url, json=payload, headers=headers)

    parsed_results = []
    for line in response.text.splitlines():
        try:
            parsed_results.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Failed to parse line: {line}, Error: {e}")

    full_response = "".join(
        result["message"]["content"] for result in parsed_results if "message" in result
    )
    return full_response


def process_message(message: str, history: list) -> str:
    agent_service_url = (
        f"http://{constants.AGENT_SERVICE_URL}:{app_config.agent_service_port}/process"
    )
    try:
        payload = {"query": history + [{"role": "user", "content": message}]}

        response = requests.post(agent_service_url, json=payload)
        response.raise_for_status()
        return response.json().get("result", "")
    except requests.RequestException as e:
        print(f"Error calling agent service: {e}")
        return "Error: Unable to process the request"
