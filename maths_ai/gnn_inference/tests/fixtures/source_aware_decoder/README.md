# Pinned decoder fixture

`pinned_outputs.json` records outputs from `JointSourceDecoder`, `SourceCandidates`,
and `ArgumentSelector` in source commit
`b8a3de7f450c2103a04f2b7a37ca85b113c45c96`. The test has no runtime Git dependency.

The capture loaded those class definitions from the pinned Git object without
importing its experimental application modules. It supplied the stored weights,
state, local vectors, library vectors, scores, and type signal. Each case records
the initial state, count logits, adapted vectors, source/local/library logits,
and GRU states for library/local/library selections at positions 0/1/2. The four
cases combine type strengths 0 and 8 with contextual residuals 0 and 0.4.

These deterministic synthetic weights establish architecture parity. They are not
a trained checkpoint or evidence of proof-search quality.
