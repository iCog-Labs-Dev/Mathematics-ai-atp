!(import! &self ../axioms/repos/PeTTaChainer/pettachainer/metta/petta_chainer)
!(import! &self ../axioms/metamath_axioms)

;   these are a list of 18 problems extracted from the book Seventy-Five Problems for Testing
;   Automatic Theorem Provers by FRANCIS JEFFRY PELLETIER 

; P1: (p→q) ↔ (¬q→¬p)

!(test
    (collapse(query 5 kb (: $prf (↔ (Implication (Premises (PropP)) (Conclusions (PropQ)))
                          (Implication (
                            Premises (Not (PropQ))) (Conclusions (Not (PropP))))) $tv)))
        ((: (no_inverse con34b)
            (↔ (Implication (Premises (PropP)) (Conclusions (PropQ))) 
            (Implication (Premises (Not (PropQ))) (Conclusions (Not (PropP))))) (STV 1.0 1.0))))


;; P2: ¬¬p ↔ p

!(test
    (collapse(query 10 kb (: $prf (↔ (PropP) (Not (Not (PropP))) ) $tv)))
        ((: (no_inverse notnotb) (↔ (PropP) (Not (Not (PropP)))) (STV 1.0 1.0))))


;; P3: ¬(p→q) → (q→p)

; !(query 10 kb (: $prf (Implication
;                   (Premises (Not (Implication (Premises (PropP)) (Conclusions (PropQ)))))
;                   (Conclusions (Implication (Premises (PropQ)) (Conclusions (PropP))))) $tv))


; P4: (¬p→q) ↔ (¬q→p)

!(test
    (collapse(query 10 kb (: $prf (↔ (Implication (Premises (Not (PropP))) (Conclusions (PropQ)))
                          (Implication (Premises (Not (PropQ))) (Conclusions (PropP)))) $tv)))
        ((: (no_inverse con1b) 
            (↔ (Implication (Premises (Not (PropP))) (Conclusions (PropQ))) 
            (Implication (Premises (Not (PropQ))) (Conclusions (PropP)))) (STV 1.0 1.0))))


; P5: ((p∨q)→(p∨r)) → (p∨(q→r))
!(query 20 kb (: $prf (Implication
                          (Premises (Implication (Premises (∨ (PropP) (PropQ))) (Conclusions (∨ (PropP) (PropR)))))
                          (Conclusions (∨ (PropP) (Implication (Premises (PropQ)) (Conclusions (PropR)))))) $tv))


;; P6: p ∨ ¬p 

!(test
    (collapse(query 10 kb (: $prf (∨ (PropP) (Not (PropP))) $tv)))
    ((: (no_inverse exmid) (∨ (PropP) (Not (PropP))) (STV 1.0 1.0))))


;; P7: p ∨ ¬¬¬p  -- "Expanded Law of Excluded Middle"

!(test
    (collapse(query 10 kb (: $prf (∨ (PropP) (Not (Not (Not (PropP))))) $tv)))
    ((: (no_inverse pm2.13) (∨ (PropP) (Not (Not (Not (PropP))))) (STV 1.0 1.0))))


; ;; P8 (Peirce's Law): ((p→q)→p)→p

; ; !(query 10 kb (: $prf (Implication (Premises (Implication
; ;                                        (Premises (Implication (Premises (PropP)) (Conclusions (PropQ))))
; ;                                        (Conclusions (PropP))))
; ;                           (Conclusions (PropP))) $tv))


; ; ;; P9: [(p∨q) & (¬p∨q) & (p∨¬q)] → ¬(¬p∨¬q)
; !(query 150 kb (: $prf (Implication
;                           (Premises (∧ (∨ (PropP) (PropQ))
;                                        (∧ (∨ (Not (PropP)) (PropQ))
;                                           (∨ (PropP) (Not (PropQ))))))
;                           (Conclusions (Not (∨ (Not (PropP)) (Not (PropQ)))))) $tv))


;; P10: argument-form problem — assert premises, then query the conclusion
; !(compileadd kb (: p10-prem1 (Implication (Premises (PropQ)) (Conclusions (PropR))) (STV 1.0 1.0)))
; !(compileadd kb (: p10-prem2 (Implication (Premises (PropR)) (Conclusions (∧ (PropP) (PropQ)))) (STV 1.0 1.0)))
; !(compileadd kb (: p10-prem3 (Implication (Premises (PropP)) (Conclusions (∨ (PropQ) (PropR)))) (STV 1.0 1.0)))
; !(query 30 kb (: $prf (↔ (PropP) (PropQ)) $tv))


;  P11: p ↔ p

 !(test
    (collapse (query 10 kb (: $prf (↔ (PropP) (PropP)) $tv)))
        ((: (no_inverse biid) (↔ (PropP) (PropP)) (STV 1.0 1.0))))


;  P12: ((p↔q)↔r) ↔ (p↔(q↔r))

 !(test
    (collapse(query 10 kb (: $prf (↔ (↔ (↔ (PropP) (PropQ)) (PropR))
                          (↔ (PropP) (↔ (PropQ) (PropR)))) $tv)))
        ((: (no_inverse biass) (↔ (↔ (↔ (PropP) (PropQ)) (PropR)) (↔ (PropP) (↔ (PropQ) (PropR)))) (STV 1.0 1.0))))


; P13 (distributive law): (p∨(q∧r)) ↔ ((p∨q)∧(p∨r))
!(test
    (collapse(query 10 kb (: $prf (↔ (∨ (PropP) (∧ (PropQ) (PropR)))
                          (∧ (∨ (PropP) (PropQ)) (∨ (PropP) (PropR)))) $tv)))
        ((: (no_inverse ordi) (↔ (∨ (PropP) (∧ (PropQ) (PropR))) (∧ (∨ (PropP) (PropQ)) (∨ (PropP) (PropR)))) (STV 1.0 1.0))))

; ;; P14: (p↔q) ↔ ((q∨¬p) ∧ (¬q∨p))
; !(query 50 kb (: $prf (↔ (↔ (PropP) (PropQ))
;                           (∧ (∨ (PropQ) (Not (PropP))) (∨ (Not (PropQ)) (PropP)))) $tv))


; P15: (p→q) ↔ (¬p∨q)
!(test
    (collapse(query 10 kb (: $prf (↔ (Implication (Premises (PropP)) (Conclusions (PropQ)))
                          (∨ (Not (PropP)) (PropQ))) $tv)))
        ((: (merge/revision (no_inverse imor) (no_inverse pm4.64)) (↔ (Implication (Premises (PropP)) (Conclusions (PropQ))) (∨ (Not (PropP)) (PropQ))) (STV 1.0 0.999950002499875))))



; P16: (p→q) ∨ (q→p)
!(test
    (collapse(query 10 kb (: $prf (∨ (Implication (Premises (PropP)) (Conclusions (PropQ)))
                          (Implication (Premises (PropQ)) (Conclusions (PropP)))) $tv)))
        ((: (merge/revision (no_inverse pm5.14) (no_inverse pm5.13)) (∨ (Implication (Premises (PropP)) (Conclusions (PropQ))) (Implication (Premises (PropQ)) (Conclusions (PropP)))) (STV 1.0 0.999950002499875))))


; ;; P17: ((p∧(q→r))→s) ↔ ((¬p∨q∨s) ∧ (¬p∨¬r∨s))
; !(query 50 kb (: $prf (↔ (Implication
;                              (Premises (∧ (PropP) (Implication (Premises (PropQ)) (Conclusions (PropR)))))
;                              (Conclusions (PropS)))
;                           (∧ (∨ (Not (PropP)) (∨ (PropQ) (PropS)))
;                              (∨ (Not (PropP)) (∨ (Not (PropR)) (PropS))))) $tv))








