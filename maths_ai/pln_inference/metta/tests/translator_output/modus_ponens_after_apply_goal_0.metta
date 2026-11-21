!(import! &self (library lib_import))

!(git-import! "https://github.com/rTreutlein/PeTTaChainer.git") 

!(import! &self (library PeTTaChainer "pettachainer/metta/petta_chainer"))

!(import! &self /home/nolawi/clone-maths/maths_ai/pln_inference/metta/axioms/metamath_axioms.metta)

; === Test: modus_ponens_after_apply ===

; === Goal index: 0 ===

; === Target formula: (P) ===



!(compileadd kb (: local-hyp-b66d4908-0 (P) (STV 1.0 1.0)))
!(compileadd kb (: local-hyp-b66d4908-1 (Implication
   (Premises (P))
   (Conclusions (Q))
) (STV 1.0 1.0)))
!(query 10 kb (: $prf (P) $tv))

; --- Cleanup ---
;; cleanup needed for local-hyp-b66d4908-0
;; cleanup needed for local-hyp-b66d4908-1