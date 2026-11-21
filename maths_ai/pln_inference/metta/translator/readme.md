# Shared typed PLN batch translator

Live scoring and batch generation use `maths_ai.pln_inference.compiler` and the
same rule-proof answer validator. See [the PLN guide](../../README.md) for setup,
supported grammar, limitations, native verification, and dataset coverage auditing.

Generate with `python -m maths_ai.pln_inference.metta.translator.translator_modules.cli`
using `--input`, `--output`, `--source-root`, and `--pantograph-repl`.
`PETTACHAINER_ROOT` selects the local fork; `--chainer-root` overrides it.
`--axioms-path` optionally overrides the fork's default axiom file.

Every supported subgoal gets a compiled MeTTa program, exact target formula, and
exit-code path in a versioned manifest. Unsupported subgoals remain in that
manifest with their diagnostic and are not sent to PeTTa. Execute the generated
shell runner before ranking; it honors `PETTA_BIN` and records process exit codes.

Rank with `--rank-manifest <generated_manifest.json>`. Missing/failed exit status,
unsupported translation, and compilation output cannot supply genuine scores or
update the Thompson sampler. Only a target-matching rule-proof and its attached
STV can produce an `ok` result. Completed no-derivation queries may receive a
separately labeled exploration score; sampled values are not truth values.

The old `--normalize-variables`, destructive binder normalization, shortened
constant names, numeric concept buckets, and permissive printed-text parser have
been removed. Regenerate old artifacts rather than reusing their identities.
