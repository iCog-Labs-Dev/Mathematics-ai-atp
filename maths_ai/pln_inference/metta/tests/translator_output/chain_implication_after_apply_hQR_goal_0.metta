!(import! &self (library lib_import))

!(git-import! "https://github.com/rTreutlein/PeTTaChainer.git") 

!(import! &self (library PeTTaChainer "pettachainer/metta/petta_chainer"))

!(import! &self /home/nolawi/clone-maths/maths_ai/pln_inference/metta/axioms/metamath_axioms.metta)

; === Test: chain_implication_after_apply_hQR ===

; === Goal index: 0 ===

; === Target formula: (Q) ===



!(compileadd kb (: local-hyp-fc658a6d-0 (P) (STV 1.0 1.0)))
!(compileadd kb (: local-hyp-fc658a6d-1 (Implication
   (Premises (P))
   (Conclusions (Q))
) (STV 1.0 1.0)))
!(compileadd kb (: local-hyp-fc658a6d-2 (Implication
   (Premises (Q))
   (Conclusions (R))
) (STV 1.0 1.0)))
!(query 10 kb (: $prf (Q) $tv))

; --- Cleanup ---
;; cleanup needed for local-hyp-fc658a6d-0
;; cleanup needed for local-hyp-fc658a6d-1
;; cleanup needed for local-hyp-fc658a6d-2