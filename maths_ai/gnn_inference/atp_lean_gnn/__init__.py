from .argument_selector import (
    ArgumentSelector,
    TacticWithArgsClassifier,
    TacticWithArgsConfig,
    compute_combined_loss,
    resolve_arg_targets_to_padded,
)
from .argument_training import (
    evaluate_model_with_args,
    train_one_epoch_with_args,
)
from .audit import DEFAULT_AUDIT_OUTPUT_ROOT, ParserAuditConfig, run_parser_audit
from .analysis import analyze_saved_run, compare_saved_runs, load_metrics_history, load_run_summary, render_run_comparison_markdown
from .cache import SplitReport, build_failure_record, build_json_payload
from .cli import DEMO_STATE
from .dataset import DatasetRow, iter_dataset_rows
from .graph import DAGBuilder, GraphNode, GraphStats, dag_to_dict, graph_stats, lemma_statement_to_dag, proof_state_to_dag, write_dag_json
from .inference import InferencePipeline
from .lemma_corpus import LemmaRecord, load_lemma_corpus, load_lemma_name_index
from .labels import (
    DEFAULT_ARITY,
    EMPTY_TACTIC,
    TACTIC_ARITY,
    UNKNOWN_TACTIC,
    build_tactic_vocab,
    encode_tactic_name,
    get_tactic_arity,
    label_example,
    normalize_tactic,
    parse_tactic_arguments,
)
from .lemma_index import LemmaIndex, LemmaIndexConfig
from .model import GraphSAGEClassifierConfig, GraphSAGEStateClassifier
from .preparation import PreparedExample, prepare_example
from .premise_pool import CandidatePool, build_unified_pools
from .premise_scoring import PremiseScorer, PremiseScorerConfig, compute_premise_ranking_loss
from .premise_training import evaluate_model_with_premises, train_one_epoch_with_premises
from .preprocess import DEFAULT_OUTPUT_ROOT, PreprocessConfig, run_preprocessing
from .pyg import NODE_TYPE_TO_ID, build_premise_mask, build_vocab, build_vocab_from_labels, dag_to_pyg
from .state import Hypothesis, ProofState, parse_state
from .training import (
    DEFAULT_BASELINE_CONFIG_PATH,
    BaselineConfig,
    PreparedGraphDataset,
    PreparedMetadata,
    TrainingLoopConfig,
    build_dataloaders,
    compute_eval_metrics_from_logits,
    evaluate_baseline_run,
    evaluate_model,
    load_baseline_config,
    load_prepared_metadata,
    train_baseline,
)
from .visualize import build_visualization_html, visualize_dag

__all__ = [
    "ArgumentSelector",
    "BaselineConfig",
    "CandidatePool",
    "DAGBuilder",
    "DEFAULT_ARITY",
    "DEFAULT_AUDIT_OUTPUT_ROOT",
    "DEFAULT_BASELINE_CONFIG_PATH",
    "DEFAULT_OUTPUT_ROOT",
    "DEMO_STATE",
    "DatasetRow",
    "EMPTY_TACTIC",
    "GraphNode",
    "GraphSAGEClassifierConfig",
    "GraphSAGEStateClassifier",
    "GraphStats",
    "Hypothesis",
    "InferencePipeline",
    "LemmaIndex",
    "LemmaIndexConfig",
    "LemmaRecord",
    "NODE_TYPE_TO_ID",
    "ParserAuditConfig",
    "PreparedExample",
    "PreparedGraphDataset",
    "PreparedMetadata",
    "PremiseScorer",
    "PremiseScorerConfig",
    "PreprocessConfig",
    "ProofState",
    "SplitReport",
    "TACTIC_ARITY",
    "TacticWithArgsClassifier",
    "TacticWithArgsConfig",
    "TrainingLoopConfig",
    "UNKNOWN_TACTIC",
    "analyze_saved_run",
    "build_dataloaders",
    "build_failure_record",
    "build_json_payload",
    "build_premise_mask",
    "build_tactic_vocab",
    "build_unified_pools",
    "build_visualization_html",
    "build_vocab",
    "build_vocab_from_labels",
    "compare_saved_runs",
    "compute_combined_loss",
    "compute_eval_metrics_from_logits",
    "compute_premise_ranking_loss",
    "dag_to_dict",
    "dag_to_pyg",
    "encode_tactic_name",
    "evaluate_baseline_run",
    "evaluate_model",
    "evaluate_model_with_args",
    "evaluate_model_with_premises",
    "get_tactic_arity",
    "graph_stats",
    "iter_dataset_rows",
    "label_example",
    "lemma_statement_to_dag",
    "load_baseline_config",
    "load_lemma_corpus",
    "load_lemma_name_index",
    "load_metrics_history",
    "load_prepared_metadata",
    "load_run_summary",
    "normalize_tactic",
    "parse_state",
    "parse_tactic_arguments",
    "prepare_example",
    "proof_state_to_dag",
    "render_run_comparison_markdown",
    "resolve_arg_targets_to_padded",
    "run_parser_audit",
    "run_preprocessing",
    "train_baseline",
    "train_one_epoch_with_args",
    "train_one_epoch_with_premises",
    "visualize_dag",
    "write_dag_json",
]

