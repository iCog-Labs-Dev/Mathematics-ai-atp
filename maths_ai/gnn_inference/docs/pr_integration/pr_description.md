# Draft PR: GNN data, retrieval, joint actions and native action coverage

## Summary

Bring the accumulated GNN engineering/research work toward main integration:
trustworthy Lean-derived graph data, reproducible training, external-premise
retrieval, complete flat-action decoding, portable model releases and native
validation of expanded structured actions.

This spans August–November work plus June/July foundations. It is broader than
the latest template patch. Implementation snapshot `13e0fdf` was 178 commits
ahead/56 behind upstream main, with 216 changed files. Reconciliation may change
those counts. This is a draft integration narrative, not a statement that the
branch is conflict-free or all runtime integrations are complete.

## Detailed report and evidence

- [Engineering report](engineering_report.md): full chronology and decisions.
- [Evidence index](evidence_index.json): artifact locations and availability.
- [Reading/attachment guide](README.md): review order, raw JSON, PDF and discussion copies.

When pasting into GitHub, replace these relative links with the corresponding
branch-pinned repository URLs. Upload/link the server artifacts identified in
the evidence index before treating undocumented raw evidence as attached.

## Earlier work included

- Lean S-expression and binder/context correctness; source-invocation extraction.
- GraphSAGE/GATv2, readout ablations, finite-loss/GPU fixes, graph-budget batching,
  packed data and resumable long-running jobs.
- Earlier premise-scorer training/handoff for RL, literature-driven GAT work and
  research scoping; no unmeasured diffusion-model result is claimed.
- Mathlib environment corpus: 318,953 declarations; repaired local/global target
  eligibility instead of selecting only benchmark theorem names.
- Published 227,276-row normalized S-expression dataset and resumable Parquet packing.
- Stop-head/GRU argument decoding, then explicit count/local-library source/identity
  prediction; repeated arguments, corrected citation/Unicode labels and frozen rankings.
- Vocabulary/encoder/index hash binding, strict portable loading and offline export checks.
- Graph/BM25/text retrieval comparisons, hard negatives, index caching and reranking audits.
- Local projection refinement and optional soft type cues; unsuccessful query/reranker
  variants documented and rejected rather than silently adopted.
- Published complete Mathlib-Sexpr-GNN v2.0.0 joint system with matched retrieval assets.
- Native source-snapshot replay, blocker census, canonical inventories and V1–V7
  typed action grammars, including cached-syntax method selectors.

## Results: keep metric types separate

| Result | Scope |
|---|---|
| 227,276 published rows | Dataset release, not action/model accuracy |
| 34.22% tactic top-1 | 4,156 known normalized val labels |
| 1,915/2,765 any-citation R@200 | Matched epoch-5 text retrieval; not all-premise recall |
| 152→169/277 local identity | Paired optional-cue offline result |
| 64→75/759 oracle nonempty exact actions | Same local weights/candidate policy; offline imitation |
| 19→27/759 predicted nonempty exact actions | Same paired cue comparison |
| 148,869/218,727 train; 2,820/4,157 val | V6 static action support: 68.06% / 67.84% |
| 41/46 native semantic transitions | Targeted 59-row probe; five rendered failures remain historical |
| V7 full support | Pending server audit; no numerical gain claimed |

Current published model is **pre-template**. Expanded target coverage has not
yet become a newly trained template-aware policy.

## Release and compatibility

- HF model: `iCogLabs-Dev/Mathlib-Sexpr-GNN`, tag `v2.0.0`, revision
  `8aba62625dfeb833e4591498700f351f3bc60af9`.
- Complete policy, vocabularies, epoch-5 text retriever/tokenizer, index and corpus.
- Preserve model/retrieval/vocabulary bindings; do not mix same-dimensional assets.
- Live normalized-state/context parity, ranked decoding and Docker/API migration
  remain tracked by [#81](https://github.com/iCog-Labs-Dev/Mathematics-ai-atp/issues/81).
- Coordinate core graph/model/action/inference changes with
  [MCTS PR #84](https://github.com/iCog-Labs-Dev/Mathematics-ai-atp/pull/84).

Related DeepSeek proof-search work was extracted to `jajos12/lean-proof-search`.
Its later 182/244 validation result is documented as related work, not a GNN
score or a feature automatically delivered by this PR.

## Verification and remaining review

- Latest focused GNN/template suite: **172 passed**, including real pinned Lean checks.
- V7 source-only native fixtures: **44/44** shared scope/method actions,
  **27/27** V5 regressions, **5/5** extra selector cases.
- Export/load-back and model/index bindings verified in their release workflow.
- Detailed historical tests/audits remain linked by component; test counts are
  not summed across different revisions as though they were one suite.
- Pending: attach server JSON evidence, obtain V7 full-cohort result, retain the
  five-row repair recheck, and run reconciled project/API integration checks.
- Inspect incidental historical changes and duplicated prover tooling during
  integration; this description is not approval of every accumulated edit.

## Reviewer reading order

1. Representation/preparation and source identity.
2. Vocabulary/checkpoint/index integrity and portable bundle loading.
3. Retriever/candidate contracts and joint argument decoding.
4. Native control/transition checks and structured action grammars.
5. Runtime adapters and main/MCTS integration boundaries.

See the engineering report for experiments that did not improve accepted
complete actions, changed-cohort qualifications, and operational failures.
