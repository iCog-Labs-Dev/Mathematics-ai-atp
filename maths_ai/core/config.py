from pathlib import Path
from dataclasses import dataclass
from typing import Optional
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

# Hybrid reasoner / inference-engine asset paths.
# Defaults mirror the paths the CLI prover (joint_inference.py) historically
# hardcoded; every one can be overridden via its environment variable (e.g. in
# Docker or by the API service) instead of assuming CLI-relative locations.
GNN_RUNS_DIR = DATA_ROOT / "maths_ai" / "gnn_inference" / "runs"

GNN_CONFIG_PATH = Path(
    os.getenv(
        "MATHS_AI_GNN_CONFIG",
        str(GNN_RUNS_DIR / "sexpr_pointer_pipeline" / "pointer_gnn" / "config.json"),
    )
)
TACTIC_MODEL_PATH = Path(
    os.getenv("MATHS_AI_TACTIC_MODEL", str(GNN_RUNS_DIR / "sexpr_pointer_pipeline" / "pointer_gnn" / "best.pt"))
)
ARGUMENT_MODEL_PATH = Path(
    os.getenv("MATHS_AI_ARGUMENT_MODEL", str(GNN_RUNS_DIR / "sexpr_pointer_pipeline" / "premise_gnn" / "best.pt"))
)
LEMMA_INDEX_PATH = Path(os.getenv("MATHS_AI_LEMMA_INDEX", str(GNN_RUNS_DIR / "lemma_index_v1")))
LEMMA_CORPUS_PATH = Path(
    os.getenv("MATHS_AI_LEMMA_CORPUS", str(GNN_RUNS_DIR / "lemma_corpus_v1" / "lemmas.jsonl"))
)

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
    gnn_config_path: Path = GNN_CONFIG_PATH
    tactic_model_path: Path = TACTIC_MODEL_PATH
    argument_model_path: Path = ARGUMENT_MODEL_PATH
    lemma_index_path: Path = LEMMA_INDEX_PATH
    lemma_corpus_path: Path = LEMMA_CORPUS_PATH
    dts_state_dir: Path = DTS_STATE_DIR
    dts_state_file: Path = DTS_STATE_FILE
    dts_default_c: float = DTS_DEFAULT_C
    dts_default_seed: int = DTS_DEFAULT_SEED
    
    # API server settings
    api_host: str = os.getenv("MATHS_AI_API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("MATHS_AI_API_PORT", "8000"))
    api_prove_timeout: Optional[float] = None  # Per-request prove timeout (None = no timeout)
    api_default_prove_timeout: float = float(os.getenv("MATHS_AI_API_PROVE_TIMEOUT", "60.0"))

settings = Settings()
