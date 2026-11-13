;; PeTTaChainer import header for generated PLN query files.
;;
;; PeTTa resolves `import!` paths relative to the directory of the top-level
;; .metta file. PLNInference therefore generates each query file inside the
;; PeTTaChainer workspace (PETTA_WORKSPACE), which links every module of the
;; vendored PeTTaChainer sources (PETTA_CHAINER_DIR, defaulted to the Docker
;; layout `/opt/pettachainer/pettachainer/metta`; see PLNInference._prepare_workspace).
;; With that layout the bare-name import below resolves to `petta_chainer.metta`
;; and pulls in its sibling modules (compile, chainer_utils, forward/backward
;; chainer, ...) the same way.
;;
;; Note: the older `(import! &self (library PeTTaChainer "<abs path>"))` form
;; CANNOT be used here — PeTTa resolves the two-argument library form against
;; `library_path/1`, which only exists after a `git-import!` (requires network
;; access at query time). Vendoring + relative import avoids both.
!(import! &self petta_chainer)
