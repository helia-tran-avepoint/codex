import os
import sys
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from shared.app_config import AppConfig, patch_config
from shared.logger import get_logger

service_name = Path(__file__).parent.name
AppConfig.load_env(service_name)
app_config = patch_config(service_name)
logger = get_logger("agent_sevice", app_config.debug)
