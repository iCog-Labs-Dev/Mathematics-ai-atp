from pathlib import Path
from dataclasses import dataclass
import os
ROOT_DIR = Path(__file__).resolve().parent.parent

# ENV = os.getenv("ENV", "development")
# DEBUG = os.getenv("DEBUG", "True").lower() == "true"

# APP_NAME = os.getenv("APP_NAME", "MyApplication")
# APP_VERSION = os.getenv("APP_VERSION", "1.0.0")



DATA_DIR = ROOT_DIR / "data"
# RAW_DATA_DIR = DATA_DIR / "raw"
# PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = ROOT_DIR / "gnn_inference" / "runs" / "premise_gnn"
CHECKPOINTS_DIR = MODELS_DIR

LOGS_DIR = ROOT_DIR / "logs"
TEMP_DIR = ROOT_DIR / "tmp"

CONFIG_DIR = ROOT_DIR / "gnn_inference" /"configs"
OUTPUT_DIR = ROOT_DIR / "outputs"
@dataclass(frozen=True)
class Settings:
    # app_name: str = "APP_NAME"
    # version: str = APP_VERSION
    # env: str = ENV
    # debug: bool = DEBUG

    root_dir: Path = ROOT_DIR
    data_dir: Path = DATA_DIR
    models_dir: Path = MODELS_DIR
    logs_dir: Path = LOGS_DIR
    proof_depth: int = 20

settings = Settings()
