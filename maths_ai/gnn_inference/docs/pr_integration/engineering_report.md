# GNN engineering and research workstream report

## 1. Scope, ownership and evidence

This report covers **August–November 2026** work on `gnn-issue-fixes`, while
retaining June/July foundations visible in its history. It describes engineering
delivery and research decisions, not a single three-month training experiment.
Implementation snapshot: **`13e0fdf`** (2026-11-07).

At that snapshot, the upstream comparison contained 178 branch-only commits,
56 upstream-only commits and 216 changed files. Git author records attribute
175 branch-only commits to Yonatan Getachew, two to Yonatan Getachew Beshah,
and one to Melkamzer Anteneh. These are Git author strings, not a claim that
every imported idea or collaborator contribution belongs to one person.
The premise-scorer handoff to Nolawi and later MCTS/API collaboration are part
of the work's integration history.

Sources include Git history, committed component documentation, historical
weekly records, server summaries supplied during development and recorded
raw-report audits. Most server artifacts are not present in this checkout.
[The evidence index](evidence_index.json) separates documented results, recorded
audit hashes, locally verified implementation checks and pending measurements.
No missing report or fresh hash is fabricated.

## 2. Executive summary

The work progressed through five connected problems:

1. Build trustworthy Lean proof-state graphs and action labels.
2. Make training, retrieval and model distribution reliable and reproducible.
3. Learn complete flat actions, not only tactic names.
4. Render structured actions and test their transitions in native Lean.
5. Expand representation coverage using cached syntax rather than repeated
   expensive source extraction.

Major deliverables include the **227,276-row normalized dataset publication**,
the **318,953-declaration Mathlib premise corpus**, GRU/count/source-aware
argument decoding, bound portable model bundles, the published **v2.0.0 joint
system**, native action replay, and V1–V7 structured action tooling.

These are distinct kinds of progress. Tactic accuracy, retrieval recall,
representation coverage, native transition parity and completed theorem proofs
must not be presented as interchangeable scores.

## 3. Chronology and earlier engineering contributions

| Period | Work and reason | Representative commits |
|---|---|---|
| June/July foundations | Mathlib discovery/pins, binder extraction, rewrite formatting, hypothesis S-expressions, de Bruijn ordering and end-to-end graph path | `e98cdbb`, `4c1099c`, `22d6daa`, `a247272`, `0291639` |
| July–early August | GATv2 option, dual-GPU execution, finite-loss checks, DataParallel/NCCL fallback, worker/shared-memory fixes, configuration handling, packed graphs and readout ablations | `8817f9a`, `ee8c1ae`, `25e9fb9`, `575e4c9`, `ae77127`, `8abf66b`, `efbcc63` |
| August extraction | Source-invocation alignment, branching/compound tactics, worker recycling, dependent-binder-safe normalized graphs and deterministic pilots | `221863c`, `3a752fa`, `fa98c23`, `c905828`, `12636db`, `cb65a1a` |
| August scale/distribution | Graph-budget batching, resume, pointer initialization, structured targets, real Mathlib corpus, row-level failure isolation, Parquet publication and portable bundles | `cb72ef7`, `c7949d4`, `e06bdb7`, `e0c3e59`, `8ba015e`, `bd4de91`, `7005e09` |
| September | Stop-head/GRU arguments, indexing and legacy-load repairs, vocabulary/encoder/index integrity, external retrieval and unified scoring | `cdb742a`, `5c97e3a`, `61453fa`, `4c5f1c3` |
| October retrieval/labels | Graph/BM25/text comparisons, hard-negative/index caching, repeated arguments, Unicode/citation repair and frozen candidate rebinding | `9c1f4f6`, `bf19af9`, `bde3ebe`, `82d7bfd`, `44a2610` |
| October joint actions | Joint local/library GRU, learned count, count ablation, local projection training and optional type cues | `6a82176`, `08aa04d`, `d4605a8`, `d00e3df`, `8f4b025` |
| Late October/November | Rejected library-query pilots, native replay, action census/templates, full joint release, semantic diagnostics and cached method selectors | `be3245f`, `57e1b27`, `2fc8650`, `ea72894`, `42c2c46`, `7414d19`, `13e0fdf` |

Complete chronology can be regenerated with:

```sh
git log --reverse --date=short --format='%ad %h %s' \
  f7e834299582d4b5adb8bfb0c484a4939169b0f1..13e0fdf
```

Commit dates establish chronology, not time spent or controlled experimental
comparability. Some commits import or merge earlier work.

## 4. Graph representation and training infrastructure

### Correctness before architecture

Earlier S-expression support could silently fall back to printed-text parsing.
Fixing that path was necessary before interpreting an S-expression experiment.
Other repairs covered multi-variable/outermost binders, de Bruijn context order,
decomposition of hypothesis types, shared free-variable identities and local lets.

The extraction approach changed from reconstructing theorem replay to capturing
the original source invocation. Branch/input-state alignment and compound tactic
capture preserved the surrounding Lean context instead of assuming that printed
goals reproduce it.

### Training and operational reliability

The branch added GATv2 alongside GraphSAGE and evaluated state-conditioned/global
readouts. It also addressed NaNs, ignored attention-head/configuration options,
multi-GPU Batch handling, NCCL fallback, explicit CUDA indices, DataLoader shared
memory pressure, graph-budget batches and oversized examples. Packed graph caches,
resume overrides, prompt logging and partial-run preservation made long runs more
practical on frequently respawning server containers.

Historical August notes report a max-readout ablation being abandoned after a
negative result. No unrecorded numerical improvement is assigned to that ablation.
They also record quarter-plan review, exploration of a proposed diffusion-based
direction, and GAT changes informed by [arXiv:2502.09263](https://arxiv.org/abs/2502.09263).
Those activities belong in the research history, but the records do not establish
a trained diffusion model or an isolated numerical gain from each proposed GAT
change. August 17 records premise-scorer training and a handoff to Nolawi for RL,
followed by development of separate premise encoders and hard-negative training.
The original handoff checkpoint/report should be attached if it is part of the
integration claim; its current raw artifact is not available in this checkout.
An earlier baseline's **35.39%** on a **2,009-row** validation cohort and the later
normalized baseline's **34.22%** on **4,156 known labels** are different cohorts;
their difference is not a measured regression.

## 5. Dataset extraction, label coverage and publication

### A missing corpus was a supervision bottleneck

The original premise pool contained theorems the benchmark proves, rather than
all declarations its tactics cite. Many frequently cited elementary lemmas had
no tactic-traced theorem row and were absent entirely.

Pantograph environment extraction produced **318,953 declarations**, including
219,626 theorems and 88,434 definitions. On the recorded **5,187-row train audit**,
the rebuilt corpus covered **9,528/9,528 citations**, all **7,572/7,572 local
references** resolved, and resolved reference-position coverage reached **96.1109%**.
The earlier broken-pointer audit had reported **9.90%**. These are historical
reference-resolution diagnostics, not full-dataset action coverage or model scores.
See the [extractor guide](../../lean_extractor/README.md).

### Extraction scale and failure isolation

A missing-row investigation distinguished unattempted files from actual errors:
historical notes found 2,033 unopened train files and theorem-wide failure
discarding otherwise recoverable rows. Shared capture reduced a planned **7,668**
file compiles to **3,602**, and individual row failures no longer automatically
voided the whole theorem. Atomic, digest-bound caches survived interruptions.

Historical train extraction grew raw S-expressions **142,632 → 221,856** and
structured V3 traces **5,188 → 218,727**. These intermediate extraction counts
are not silently substituted for the later prepared cohort.

### Published dataset and packaging

The August publication record identifies
[Mathlib-Normalized-Sexpr](https://huggingface.co/datasets/jajostrains/Mathlib-Normalized-Sexpr)
with **227,276 rows**. Packing roughly 680k per-target JSON files into resumable
Parquet shards made distribution practical. License notices from Mathlib and
LeanDojo were retained. A stale 3,000-row extraction manifest was prevented from
being misreported as the full train split's failure taxonomy.

Later normalized prepared population: **218,727 train / 4,157 val / 4,392 test**.
Those sum to 227,276, but publication and preparation remain separately bound
artifacts. Validation drove model/coverage decisions; historical deterministic
test preprocessing is not an official held-out proving result.

## 6. Portable models and artifact binding

A training checkpoint originally depended on external vocabularies named by a
host filesystem path. Same-sized but differently ordered vocabularies could load
while changing every label's meaning. Bundles now include vocabulary content and
hashes, explicit architecture/configuration and strict or deliberately bounded
partial loading. Offline load tests remove the prepared dataset to expose hidden
dependencies. Published manifests remove training-machine paths.

The same integrity problem existed for lemma vectors: equal dimensionality does
not mean two encoders share an embedding space. Index loading now binds encoder
weights, normalization and ID mappings, and rejects mismatches.

The full [joint release](../huggingface_joint_release.md) subsequently combined
the refined policy with its matched epoch-5 text retriever, tokenizer, vectors,
FAISS rows and corpus. **v2.0.0** is pinned to HF revision
`8aba62625dfeb833e4591498700f351f3bc60af9`, with:

- **3,869,544 policy parameters**;
- **45,426,432 retriever parameters**;
- **318,953 indexed lemmas**.

Export/load-back passed on the server. Earlier v1.0.0 remains available at
`700c48cf1b741e59d39fa29d105cd2d2f77fdb9f`. The release does not imply that
`main`'s live API already loads or feeds the complete joint system correctly.

## 7. Retrieval experiments and what they changed

The work implemented contrastive external-premise retrieval, hard negatives,
approximate mining/index caches, resumable full-corpus indexing and candidate
rebinding. It measured graph retrieval, BM25, hybrids and text dual encoders.

| Historical comparison on 2,501 labeled val steps | Any-citation R@200 |
|---|---:|
| Frozen-lemma graph retriever, 30 epochs | 8.20% |
| BM25 using graph-label queries | 39.54% |
| BM25 using original state text | 41.30% |
| Joint graph-tower pilot, epoch 3 | 1.84% |
| Capped text pilot, epoch 8 | 55.30% |
| Full-data text run, epoch 3 | 64.81% |

These historical comparisons used recorded common labels, but different training
budgets; the table is not a matched-compute architecture benchmark. A tested
fixed-200 hybrid added only 11 hits over BM25, and rank fusion was worse. Larger
BM25 K exposed retrieval headroom rather than top-200 downstream performance.

Later repaired labels changed denominators. The matched epoch-5 retriever on the
normalized cohort hit **1,915/2,765 = 69.26%** any-citation R@200. It means at least
one recorded citation was retrieved, not every argument or a solved theorem.
See [retriever protocols](../retriever_experiments.md) and
[fixed-candidate workflow](../text_scorer_workflow.md).

## 8. Argument decoding, supervision repair and accepted local improvements

### From static arity to sequential and count-aware decoding

The original argument path used static arities and only the immediately previous
argument. The branch introduced a GRU, teacher-forced prefixes and a learned stop
head; fixed indexing, masking and legacy loads; and allowed valid repeated
references. A later audit showed stop-after-one collapse even with the correct
first argument, motivating an explicit **0–3 argument count head**.

The joint decoder learns tactic-conditioned count, local/library source and
within-source identity with recurrent feedback. Frozen text vectors enter via
an adapter; unrelated graph/text embedding spaces are not interchanged.

### Label repair was essential

Citation markup created phantom `a` arguments and ASCII parsing split Unicode
names. [Versioned relabeling](../repaired_argument_workflow.md) preserves Unicode,
uses source context/declaration identities, separates known citations from
supported flat actions and excludes unsupported argument supervision instead of
turning it into false zero-argument targets. Existing rankings could be rebound
without re-encoding unchanged queries.

Normalized flat-label population: **61,853 train / 1,207 val** supported actions,
including **448** zero-argument val actions and **759** nonempty ones. This label
policy differs from the later stricter executable flat renderer.

### Local ranking and honest acceptance criteria

Local-projection-only training on 16,816 eligible train positions improved matched
local top-1 **130 → 152/277**, with GNN/retriever/GRU fixed. Optional, train-selected
soft cues for exact/rw/simp/subst produced the following paired offline results:

| Metric, same epoch-8 weights | Cues off | Accepted optional cues |
|---|---:|---:|
| Matched local identity | 152/277 | 169/277 |
| Nonempty recorded-action exact, oracle tactic | 64/759 | 75/759 |
| Nonempty recorded-action exact, predicted tactic | 19/759 | 27/759 |

These are offline imitation metrics. The cues remain default-off unless their
matching normalized input profile is selected. [Local probe](../local_identity_probe.md)
and [joint decoder protocol](../joint_source_flat_pilot.md) document cohorts.

Negative results are part of the deliverable: inverse count weighting improved
rare count classes but harmed complete actions; a library-only reranker lost to
frozen text; three independent library-query pilots improved some ranking scores
without improving accepted predicted complete actions. The independent query
experiment was removed in `be3245f`. No best-looking intermediate score is
substituted for the accepted model.

## 9. Native replay and structured action coverage

Replay restores the original Lean environment and all active goals. Captured
syntax is the original control; source-text replay is a separate diagnostic.
Input parity, original transition, rendered transition and exact dataset
alignment are distinct counters. Failed transitions now retain normalized goal
diagnostics. Proof obligations are not reported as completed proofs.

| Renderer | Train static support | Val static support |
|---|---:|---:|
| Strict flat executable baseline | 43,864 | 862 |
| V1: directions, only, locations | 66,904 | 1,322 |
| V2: applied rules/reference lookup | 78,324 | 1,531 |
| V3: introductions/named args/long lists | 93,113 | 1,823 |
| V4: recursive terms/constructors/holes/projections | 107,913 | 2,096 |
| V5: wrappers/modifiers/declarations | 127,245 | 2,438 |
| V6: scopes and branching patterns | 148,869 | 2,820 |
| V7: cached method selector labels | Pending | Pending |

V6 is **68.06% train / 67.84% val** on the full prepared denominator. Exact-aligned
support is **138,763 / 2,815**. Original invalid traces remain quarantined
(**831 / 9**). This is representation support, not model prediction accuracy.

Native evidence includes V5/V6 real-Mathlib delta pilots passing **24/24** each.
The broader V3 val check produced **1,812/1,818** matches, with six original-control
failures retained for diagnosis. A later train-form run was stopped at user request.

The V6 semantic probe selected 59 val actions in 16 files: 54 captures bound,
46 rendered statically and **41/46** matched native transitions. Five capture
failures were ENOSPC; eight bound exclusions were two multi-tactic wrappers,
three metavariables and three lossy lambdas. Four rendered execution failures
and one executed mismatch exposed dropped instances, named arguments, types
and expanded arithmetic. Source-preserving repairs are implemented; the real
five-row recheck has not been supplied here.

V7 represents receiver plus selector label directly from cached V3 syntax,
without V2 extraction or invented global premise IDs. **172 focused tests** passed,
including **44/44** shared source-only native actions, **27/27** V5 regressions
and **5/5** extra selector cases. Its full audit is pending. See
[V7 workflow](../tactic_templates_v7.md),
[semantic probe/repairs](../semantic_recovery_extraction.md) and
[native protocol](../native_gold_replay.md).

## 10. Related proof-model/search investigation

This is a separate workstream from GNN model accuracy and Nolawi's MCTS system.
It studied tokenizer/backend fidelity, automation controls, model proof generation,
compiler feedback, frozen propositions, allowed-axiom checking, native state
search, duplicate rejection and retention of verified prefixes.

The corrected ten-task smoke gave automation **10/10** and saved model **9/10**,
so it established compatibility rather than extra model coverage. A pinned
25-task miniF2F validation development cohort progressed **8/25 → 12/25** under
documented protocol changes. Initial native searches added no solves and exposed
proposal-format/context/prefix bugs; those negative results guided repairs.

The prover was extracted into [lean-proof-search](https://github.com/jajos12/lean-proof-search),
initial commit `4e93ecd` from ATP `3d996a3`. Later work in that repository loaded
all 244 validation problems. Its separately audited development-port result was
**159/244** automation-plus-one-shot baseline, then **182/244** with 23 extra
search proofs. The disjoint outcome sets were **57 automation / 102 one-shot /
23 search**, with four input failures retained. Additional search used 1,059
completed generations and 1,450,808 output tokens; the report's SHA-256 was
recorded as `d493edb7bb9695a587fd6e6af54b6418d276326c95490ed62e14e296139a76a6`.

Those are separate-repository results under Lean-version porting, added compute
and documented crash recovery. No GNN was active; they are neither this PR's GNN
proof score nor an official test leaderboard result. The historical audit did
not independently recompile the complete final corpus locally. Earlier ATP
protocol/results are in [miniF2F development](../minif2f_development.md).

## 11. What the PR should and should not claim

Delivered: extraction/representation fixes, training infrastructure, dataset and
model packaging, retrieval and joint decoding experiments, accepted local
improvements, native validation and structured-action coverage tooling.

Still pending: V7 server gains, the real five-row source-preserving repair
recheck, broader native validity, template-aware supervision export/training,
and complete live API graph/context/ranked-action parity. The published model
remains pre-template. The requested >50k additional train gain from the latest
coverage batch has not been established.

[Issue #81](https://github.com/iCog-Labs-Dev/Mathematics-ai-atp/issues/81) describes
selective API integration, not a blanket claim that merging research code enables
v2 serving. [MCTS PR #84](https://github.com/iCog-Labs-Dev/Mathematics-ai-atp/pull/84)
overlaps core interfaces. Reconcile graph/model/checkpoint/action contracts and
merge order with its author; preserve the standalone prover's separate ownership.

The 172-test figure describes the latest focused GNN/template suite, not a fresh
exhaustive validation of every file in this large branch. A merge-ready integration
needs project/API tests against the reconciled main/MCTS contracts.

## 12. Review and evidence handoff

Start with [the PR description](pr_description.md), then use
[the evidence index](evidence_index.json) to obtain raw artifacts. Attach small
machine-readable summaries to the PR and put large raw JSON/JSONL, checkpoints
and datasets in immutable artifact storage with hashes. A PDF/Google Doc can
serve readers, while this versioned report remains the canonical narrative.
