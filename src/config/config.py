from dataclasses import dataclass, field
from typing import List, Dict, Any
import os
import yaml

@dataclass
class EdgeConfig:
    node_id: str = "edge_node_01"
    rtsp_streams: List[Dict[str, Any]] = field(default_factory=list)
    local_db_path: str = "/data/edge_cache.db"
    models_dir: str = "/app/models"
    log_level: str = "INFO"

@dataclass
class CloudConfig:
    api_url: str = "https://api.retail-ai.com"
    auth_token: str = ""
    s3_bucket: str = "retail-ai-clips"

@dataclass
class AppConfig:
    edge: EdgeConfig = field(default_factory=EdgeConfig)
    cloud: CloudConfig = field(default_factory=CloudConfig)

def load_config(config_path: str = "configs/edge_config.yaml") -> AppConfig:
    """Loads configuration parameters from file and merges environment variables."""
    config_dict: Dict[str, Any] = {}
    
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            file_data = yaml.safe_load(f)
            if file_data:
                config_dict = file_data
                
    edge_data = config_dict.get("edge", {})
    cloud_data = config_dict.get("cloud", {})
    
    # Merge environment variables
    node_id = os.getenv("EDGE_NODE_ID", edge_data.get("node_id", "edge_node_01"))
    local_db = os.getenv("EDGE_LOCAL_DB", edge_data.get("local_db_path", "/data/edge_cache.db"))
    api_url = os.getenv("CLOUD_API_URL", cloud_data.get("api_url", "https://api.retail-ai.com"))
    auth_token = os.getenv("CLOUD_AUTH_TOKEN", cloud_data.get("auth_token", ""))
    
    return AppConfig(
        edge=EdgeConfig(
            node_id=node_id,
            rtsp_streams=edge_data.get("rtsp_streams", []),
            local_db_path=local_db,
            models_dir=edge_data.get("models_dir", "/app/models"),
            log_level=edge_data.get("log_level", "INFO")
        ),
        cloud=CloudConfig(
            api_url=api_url,
            auth_token=auth_token,
            s3_bucket=cloud_data.get("s3_bucket", "retail-ai-clips")
        )
    )
