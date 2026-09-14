from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


from maths_ai.gnn_inference.atp_lean_gnn.lemma_corpus import LemmaRecord, write_lemma_corpus


DEFAULT_OUTPUT_DIR = Path("artifacts") / "lemmas" / "v1" / "corpus"


@dataclass(frozen=True)
class CorpusBuildResult:
    total_count: int
    success_count: int
    failure_count: int
    deduped_count: int


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                yield line_index, json.loads(raw)
            except json.JSONDecodeError as exc:
                yield line_index, {"_error": f"invalid_json: {exc}"}


def _normalize_record(payload: dict[str, object]) -> tuple[str, str, str, str] | None:
    name = str(payload.get("name", "")).strip()
    statement = str(payload.get("statement") or payload.get("type") or "").strip()
    namespace = str(payload.get("namespace", "")).strip()
    module = str(payload.get("module", "")).strip()

    if not name or not statement:
        return None

    if not namespace and "." in name:
        namespace = name.rsplit(".", 1)[0]
    return name, statement, namespace, module


def build_corpus(
    input_jsonl: Path,
    *,
    output_dir: Path,
    sample_limit: int | None,
    dedupe: bool,
    force: bool,
) -> CorpusBuildResult:
    if not input_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL file not found: '{input_jsonl}'.")

    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "lemmas.jsonl"
    failures_path = output_dir / "failures.jsonl"
    manifest_path = output_dir / "manifest.json"

    if corpus_path.exists() and not force:
        raise FileExistsError(
            f"Output '{corpus_path}' already exists. Use --force to overwrite."
        )

    records: list[LemmaRecord] = []
    failures: list[dict[str, object]] = []
    seen_names: set[str] = set()
    total_count = 0
    deduped_count = 0

    for line_index, payload in _read_jsonl(input_jsonl):
        if sample_limit is not None and total_count >= sample_limit:
            break
        total_count += 1

        if "_error" in payload:
            failures.append({"line_index": line_index, "reason": payload["_error"]})
            continue

        normalized = _normalize_record(payload)
        if normalized is None:
            failures.append({"line_index": line_index, "reason": "missing_name_or_statement"})
            continue

        name, statement, namespace, module = normalized
        if dedupe and name in seen_names:
            deduped_count += 1
            continue

        lemma_id = len(records)
        records.append(
            LemmaRecord(
                lemma_id=lemma_id,
                name=name,
                statement=statement,
                namespace=namespace,
                module=module,
            )
        )
        seen_names.add(name)

    write_lemma_corpus(corpus_path, records)

    if failures:
        with failures_path.open("w", encoding="utf-8") as handle:
            for failure in failures:
                handle.write(json.dumps(failure, ensure_ascii=False, sort_keys=True))
                handle.write("\n")

    manifest = {
        "input_jsonl": str(input_jsonl),
        "output_dir": str(output_dir),
        "total_count": total_count,
        "success_count": len(records),
        "failure_count": len(failures),
        "deduped_count": deduped_count,
        "sample_limit": sample_limit,
        "dedupe": dedupe,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    return CorpusBuildResult(
        total_count=total_count,
        success_count=len(records),
        failure_count=len(failures),
        deduped_count=deduped_count,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a normalized lemma corpus JSONL.")
    parser.add_argument("--input-jsonl", type=str, required=True, help="Source JSONL with lemma records")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--sample-limit", type=int, default=None, help="Optional cap on processed rows")
    parser.add_argument("--dedupe", action="store_true", help="Deduplicate by lemma name")
    parser.add_argument("--force", action="store_true", help="Overwrite existing corpus artifacts")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        result = build_corpus(
            Path(args.input_jsonl),
            output_dir=Path(args.output_dir),
            sample_limit=args.sample_limit,
            dedupe=args.dedupe,
            force=args.force,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print(
        "Built lemma corpus: "
        f"total={result.total_count}, "
        f"success={result.success_count}, "
        f"failed={result.failure_count}, "
        f"deduped={result.deduped_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
