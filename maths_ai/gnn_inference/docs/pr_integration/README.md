# GNN workstream: PR review and evidence package

This package documents the accumulated work, not only the latest template changes.
Main reporting period: **August–November 2026**, with June/July foundations retained.
Implementation snapshot: **`13e0fdf`**. Documentation commits after that snapshot
do not imply new training or evaluation results.

## Reading order

1. [PR description](pr_description.md): short, paste-ready integration narrative.
2. [Engineering report](engineering_report.md): chronology, earlier contributions,
   experiments, decisions, results, limitations and integration boundaries.
3. [Evidence index](evidence_index.json): machine-readable artifact locations,
   result provenance, known hashes and missing attachments.
4. Follow the report's links to component-specific commands and protocols.

At the implementation snapshot, GitHub comparison against upstream `main`
reported **178 ahead / 56 behind**, **216 changed files**, approximately
**47,715 additions / 1,256 deletions**. These are historical snapshot numbers,
not an assertion about the eventual integration PR after conflict resolution.

## Where reports and attachments belong

| Material | Recommended location |
|---|---|
| Review narrative, methodology, small result tables | Versioned Markdown in this package |
| Compact metric/evidence registry | `evidence_index.json` in Git |
| Raw JSON reports and JSONL per-row diagnostics | Immutable release attachment, HF dataset revision, or team artifact storage |
| Checkpoints, vectors, FAISS indexes, packed datasets | Versioned HF/artifact release, linked by revision and hashes |
| PDF | Export of the engineering report for reading/sharing |
| Google Doc | Optional discussion copy linked to the source commit; accepted corrections return to Markdown |

Keep one canonical report rather than maintaining divergent GitHub, PDF and
Google Doc versions. For a published attachment, record its URL, byte count,
SHA-256, producing code revision, cohort/environment pins and related evidence
ID. Do not substitute screenshots for machine-readable reports.

An optional local PDF export, if Pandoc and a PDF engine are installed:

```sh
pandoc maths_ai/gnn_inference/docs/pr_integration/engineering_report.md \
  --standalone --toc -o /tmp/gnn-engineering-report.pdf
```

This documentation pass creates the Markdown/JSON sources. It does not create
a Google Doc, upload an attachment, or claim a PDF was generated.

## Attachment gaps before final review

Most recent raw training/evaluation reports reside on the server, not this
checkout. Their locations are indexed, but bytes and new hashes have not been
invented. In particular, attach the current model/candidate metrics, completed
V1–V6 coverage reports, semantic probe/replay reports, and V7 coverage report
when available. Preserve complete reports behind the compact summaries.

Historical weekly notes informed this report but are locally excluded from Git.
The engineering report promotes their substantive findings into committed
documentation; reviewers do not need access to the private weekly directory.
Evidence entries identify where a finding is historical rather than a raw
report independently re-audited in this documentation pass.

## Review coordination

- [MCTS integration PR #84](https://github.com/iCog-Labs-Dev/Mathematics-ai-atp/pull/84)
  overlaps graph/model/inference interfaces. Coordinate contracts and merge order.
- [Joint-model API integration issue #81](https://github.com/iCog-Labs-Dev/Mathematics-ai-atp/issues/81)
  specifies live input parity, complete loading, ranked decoding and deployment.
- This package is a draft PR companion. No PR is opened by adding these files.
