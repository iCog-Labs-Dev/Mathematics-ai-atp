from pathlib import Path
from dataclasses import dataclass
import os

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Config directory - defined early as it's used in default paths below
CONFIG_DIR = ROOT_DIR / "maths_ai" / "config"

# ENV = os.getenv("ENV", "development")
# DEBUG = os.getenv("DEBUG", "True").lower() == "true"

# APP_NAME = os.getenv("APP_NAME", "MyApplication")
# APP_VERSION = os.getenv("APP_VERSION", "1.0.0")

# Data root - can be overridden via environment variable (e.g., in Docker)
DATA_ROOT = Path(os.getenv("MATHS_AI_DATA_ROOT", ROOT_DIR))

# Config paths for model and corpus - can be overridden via environment variables
MODEL_CONFIG_PATH = Path(os.getenv("MATHS_AI_MODEL_CONFIG", str(CONFIG_DIR / "models" / "premise_gnn.yaml")))
CORPUS_CONFIG_PATH = Path(os.getenv("MATHS_AI_CORPUS_CONFIG", str(CONFIG_DIR / "corpus" / "lemma_corpus_v1.yaml")))

# Lean project path - can be overridden via environment variable
LEAN_PROJECT_PATH = Path(os.getenv("MATHS_AI_LEAN_PROJECT", str(ROOT_DIR / "lean_project")))

DATA_DIR = ROOT_DIR / "data"
# RAW_DATA_DIR = DATA_DIR / "raw"
# PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Existing hardcoded paths stay as fallback defaults
MODELS_DIR = DATA_ROOT / "gnn_inference" / "runs" / "premise_gnn"
CHECKPOINTS_DIR = MODELS_DIR

LOGS_DIR = ROOT_DIR / "logs"
TEMP_DIR = ROOT_DIR / "tmp"

OUTPUT_DIR = Path(os.getenv("MATHS_AI_OUTPUT_DIR", str(ROOT_DIR / "outputs")))
DTS_STATE_DIR = ROOT_DIR / "dts_state"
DTS_STATE_FILE = DTS_STATE_DIR / "thompson_sampler_state.json"
DTS_DEFAULT_C = 100.0
DTS_DEFAULT_SEED = None
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
    dts_state_dir: Path = DTS_STATE_DIR
    dts_state_file: Path = DTS_STATE_FILE
    dts_default_c: float = DTS_DEFAULT_C
    dts_default_seed: int = DTS_DEFAULT_SEED

settings = Settings()
