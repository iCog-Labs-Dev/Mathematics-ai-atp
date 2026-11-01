"""Public exports are resolved on demand so graph contracts stay lightweight."""

from importlib import import_module

_EXPORT_GROUPS = {
    "argument_selector": (
        "TacticWithArgsClassifier",
        "supervised_loss",
    ),
    "source_aware_decoder": ("SourceAwareDecoder",),
    "policy_actions": ("ArgumentReference", "CandidatePoolSnapshot", "OfflineCandidateProvider", "local_pools"),
    "architectures": (
        "EncoderOutput",
        "GATv2Encoder",
        "GraphSAGEEncoder",
        "StateGraphEncoder",
    ),
    "argument_training": (
        "evaluate_model_with_args",
        "train_one_epoch_with_args",
    ),
    "audit": (
        "DEFAULT_AUDIT_OUTPUT_ROOT",
        "ParserAuditConfig",
        "run_parser_audit",
    ),
    "analysis": (
        "analyze_saved_run",
        "compare_saved_runs",
        "load_metrics_history",
        "load_run_summary",
        "render_run_comparison_markdown",
    ),
    "cache": (
        "SplitReport",
        "build_failure_record",
        "build_json_payload",
    ),
    "cli": ("DEMO_STATE",),
    "dataset": (
        "DatasetRow",
        "iter_dataset_rows",
    ),
    "graph": (
        "DAGBuilder",
        "GraphNode",
        "GraphStats",
        "dag_to_dict",
        "get_node_labels",
        "graph_stats",
        "lemma_statement_to_dag",
        "proof_state_to_dag",
        "sexp_to_dag",
        "write_dag_json",
    ),
    "inference": ("InferencePipeline",),
    "lemma_corpus": (
        "LemmaRecord",
        "load_lemma_corpus",
        "load_lemma_name_index",
    ),
    "labels": (
        "DEFAULT_ARITY",
        "EMPTY_TACTIC",
        "TACTIC_ARITY",
        "UNKNOWN_TACTIC",
        "build_tactic_vocab",
        "encode_tactic_name",
        "get_tactic_arity",
        "label_example",
        "normalize_tactic",
        "parse_tactic_arguments",
    ),
    "lemma_index": (
        "LemmaIndex",
        "LemmaIndexConfig",
    ),
    "model": ("SupervisedTacticClassifier",),
    "model_spec": ("ModelSpec",),
    "preparation": (
        "PreparedExample",
        "prepare_example",
    ),
    "premise_pool": (
        "CandidatePool",
        "build_unified_pools",
    ),
    "premise_scoring": (
        "PremiseScorer",
        "PremiseScorerConfig",
        "compute_premise_ranking_loss",
    ),
    "premise_training": (
        "evaluate_model_with_premises",
        "train_one_epoch_with_premises",
    ),
    "preprocess": (
        "DEFAULT_OUTPUT_ROOT",
        "PreprocessConfig",
        "run_preprocessing",
    ),
    "pyg": (
        "NODE_TYPE_TO_ID",
        "build_premise_mask",
        "build_vocab",
        "build_vocab_from_labels",
        "dag_to_pyg",
    ),
    "state": (
        "Hypothesis",
        "ProofState",
        "parse_state",
    ),
    "training": (
        "DEFAULT_BASELINE_CONFIG_PATH",
        "BaselineConfig",
        "PreparedGraphDataset",
        "PreparedMetadata",
        "TrainingLoopConfig",
        "build_dataloaders",
        "compute_eval_metrics_from_logits",
        "evaluate_baseline_run",
        "evaluate_model",
        "load_baseline_config",
        "load_prepared_metadata",
        "train_baseline",
    ),
    "visualize": (
        "build_visualization_html",
        "visualize_dag",
    ),
}
_EXPORT_MODULES = {
    name: module for module, names in _EXPORT_GROUPS.items() for name in names
}


def __getattr__(name: str):
    module = _EXPORT_MODULES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))


__all__ = sorted(_EXPORT_MODULES)
