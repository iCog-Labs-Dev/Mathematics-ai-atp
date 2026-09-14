!(import! &self (library lib_import))

!(git-import! "https://github.com/rTreutlein/PeTTaChainer.git") 

!(import! &self (library PeTTaChainer "pettachainer/metta/petta_chainer"))

!(import! &self /home/nolawi/clone-maths/maths_ai/pln_inference/metta/axioms/metamath_axioms.metta)

; === Test: and_commutativity_after_constructor ===

; === Goal index: 0 ===

; === Target formula: (Q) ===



!(compileadd kb (: local-hyp-5d45b6fa-0 (P) (STV 1.0 1.0)))
!(compileadd kb (: local-hyp-5d45b6fa-1 (Q) (STV 1.0 1.0)))
!(query 10 kb (: $prf (Q) $tv))

; --- Cleanup ---
;; cleanup needed for local-hyp-5d45b6fa-0
;; cleanup needed for local-hyp-5d45b6fa-1