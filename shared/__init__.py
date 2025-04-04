import logging
import os
import sys
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

logging.getLogger("watchdog").setLevel(logging.ERROR)
logging.getLogger("watchfiles").setLevel(logging.ERROR)
from shared.app_config import AppConfig, patch_config

service_name = Path(__file__).parent.name
AppConfig.load_env(service_name)
app_config = patch_config(service_name)
