import __init__
from utils import generate_service_port

print("generate & checking service port...")

all_service = [
    "index_service",
    "agent_service",
    "web_service",
    "analysis_service",
    "websocket_service",
]

for service in all_service:
    generate_service_port(service)


print("init service port done...")
