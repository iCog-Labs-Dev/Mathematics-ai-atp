# Structured action codec and pilot compilation

## Scope

`atp_lean_gnn/structured_codec.py` implements `typed_action_prefix_v1` for the
four schemas actually emitted by the server pilot:

| Schema | Train | Val |
| --- | ---: | ---: |
| `flat_reference_v2` | 3,471 | 248 |
| `intro_binders_v1` | 122 | 18 |
| `math_notation_pattern_v1` | 303 | 61 |
| `math_notation_tactic_v1` | 1,104 | 173 |

Source-program/deferred-name targets remain outside this pilot. The codec
rehydrates the existing typed dataclasses, validates fields and derived metadata,
and reuses their renderers. It does not replace typed Lean structure with an
unconstrained text/JSON prediction target.

## Operation contract

An action begins with `SCHEMA:<schema>`, then fixed-field typed products
(`OBJ:<type>`), list markers, enums, optional values, and `END_ACTION`.
There are no arbitrary predicted object keys.

- `LOCAL`: input **Lean context index** payload, not a graph node index.
- `LIBRARY`: matched corpus **lemma ID**, not a retrieved-pool position or a
  generated declaration name.
- `BOUND`: lexical declaration-slot identity plus its legal lexical scope.
- `DECLARE_LEXICAL`, `DECLARE_PATTERN`, `DECLARE_INTRO`: separate declaration
  roles with slot metadata. They do not point at input premises.
- `DECLARE_TACTIC`: a newly written tactic/equation name, not a premise.
- `TEXT`, fixed `BYTE:00`–`BYTE:ff`, `END_TEXT`: UTF-8 payloads for declaration
  names, numeric literals, method selectors and parameter labels.

Lexical/pattern slot numbering is normalized to deterministic declaration order.
**Binder spellings are preserved for the first pilot.** They are supervised name
payloads associated with typed declarations, while bound uses refer to slots.
This avoids changing native successor-state names through a new alpha-renaming
policy before the decoder pilot. Source and decoded Lean text must match exactly.

Scope checks reject unknown input locals, unknown/out-of-scope lexical references,
captured input/outer lexical references, duplicate declaration identities, unused
domain children, forged masks, invalid UTF-8 and noncanonical operation streams.
Dependent binder domains see only earlier binders; bounded-quantifier predicates
see their own binder. Existing original-control comparisons are unchanged.

Defaults: **4,096 operations**, **1,024 bytes per text field**, bounded structural
depth. Over-budget targets are explicit exclusions; targets are never truncated.
The packed token sequence includes one additional BOS token.

## Vocabulary and dynamic targets

`atp_lean_gnn/structured_vocabulary.py` fits structural tokens on **train only**.
PAD/BOS/UNK and all 256 byte tokens are fixed beforehand. New validation name
spellings need no vocabulary extension. Validation-only structural tokens are
explicit OOV exclusions, not silently learned or replaced with a valid target.

Packed records contain aligned `token_ids`, `pointer_targets`,
`declaration_slots`, `lexical_scopes` and `input_local_scopes`. Pointer values retain semantic
identities; the training loader must still map them to actual GNN/retrieval/
lexical candidate positions. Declaration IDs are metadata rather than a new
input-premise pointer loss. Shadowed lexical slots and captured/inaccessible input
locals are absent from the corresponding masks. Every pointer category carries
both scope masks, so availability does not reveal the gold local/library choice.
The validation denominator includes all candidate
rows, including codec/vocabulary exclusions.

## Compiler

`scripts/compile_structured_supervision.py` consumes:

- The completed 5,000/500 candidate package.
- Its complete native replay report, bound to exactly the prepared validation
  row selection and cache hashes, with all original/input/rendered checks passing.
- The same immutable source caches and matched corpus/index identity tables.

It verifies candidate and source hashes, performs encode/decode/render round-trips,
fits the vocabulary on train records, and writes a new output directory:

- `train.jsonl`, `val.jsonl`: operations, packed IDs, source/canonical targets,
  reference identities and original cache bindings.
- `vocabulary.json`: fixed alphabet and train-fitted structural vocabulary.
- `exclusions.jsonl`: explicit codec/budget/OOV exclusions.
- Per-split row selections and `report.json`: source/code hashes, counts, schema
  totals, length statistics, vocabulary digest and native codec gate.

`native_codec_gate_passed` requires exact rendered-text reproduction for every
native-validated pilot row. Only those validation records become
`native_validation="matched"`; unmeasured train/validation rows remain unmeasured.
The compiler reuses native evidence; it does not run Lean over every training row.

`neural_target_compiled=true` means token/payload targets exist, **not that a model
has trained**. `ready_for_training=false` and
`candidate_pool_positions_aligned=false` explicitly retain the next tasks:
retrieval-aligned loader, structured decoder/loss, and training/evaluation loop.
The accepted GNN/retriever weights are not changed by this compilation.

## Verification

Combined focused regression suite: **117 tests pass**, including 27 new codec/
compiler checks. A native fixture passes **25/25 original/generated transitions**
across all four schemas, including Unicode/anonymous introductions, dependent
binders, shadowing, equation/tuple patterns, selectors, explicit holes, bounded
quantifiers, existentials and two-goal constructor transitions.

This supplements the server's **24/24** native pilot, which contained no counted
introduction or method-selector/math-notation features. Fixture results and source
round-trips do not establish learned generation accuracy, all-row native validity,
or premise-retrieval reachability. The actual server compilation and new decoder
training remain to be run.
