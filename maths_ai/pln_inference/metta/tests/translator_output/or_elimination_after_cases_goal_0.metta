!(import! &self (library lib_import))

!(git-import! "https://github.com/rTreutlein/PeTTaChainer.git") 

!(import! &self (library PeTTaChainer "pettachainer/metta/petta_chainer"))

!(import! &self /home/nolawi/clone-maths/maths_ai/pln_inference/metta/axioms/metamath_axioms.metta)

; === Test: or_elimination_after_cases ===

; === Goal index: 0 ===

; === Target formula: (R) ===



!(compileadd kb (: local-hyp-5722b539-0 (Implication
   (Premises (P))
   (Conclusions (R))
) (STV 1.0 1.0)))
!(compileadd kb (: local-hyp-5722b539-1 (Implication
   (Premises (Q))
   (Conclusions (R))
) (STV 1.0 1.0)))
!(compileadd kb (: local-hyp-5722b539-2 (P) (STV 1.0 1.0)))
!(query 10 kb (: $prf (R) $tv))

; --- Cleanup ---
;; cleanup needed for local-hyp-5722b539-0
;; cleanup needed for local-hyp-5722b539-1
;; cleanup needed for local-hyp-5722b539-2