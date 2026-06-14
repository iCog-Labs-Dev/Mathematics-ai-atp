!(import! &self ../axioms/repos/PeTTaChainer/pettachainer/metta/petta_chainer)



; ;; test for Chained inference (P → Q → R)
    ; fact
; !(compileadd kb (: fact-P (PropP) (STV 1.0 1.0)))
; !(compileadd kb (:  P-impl-Q (Implication (Premises (PropP)) (Conclusions (PropQ))) (STV 1.0 1.0)))
; !(compileadd kb (: Q-impl-R (Implication (Premises (PropQ)) (Conclusions (PropR))) (STV 1.0 1.0)))
    ;test
    ;     !(test
    ;     (collapse(query 20 kb (: $prf (PropR) $tv)))
    ;     ((: (rule-proof Q-impl-R (rule-proof P-impl-Q fact-P)) (PropR) (STV 1.0 1.0))))

   

        ;     ;; Test for  P, Q ⊢ P ∧ Q
        ;        ; facts
    ; !(compileadd kb (: fact-P (PropP) (STV 1.0 1.0)))
    ; !(compileadd kb (: fact-Q (PropQ) (STV 1.0 1.0)))
        ;           ; test
        ;     !(test
        ;     (collapse(query 100 kb (: $prf (And (PropP) (PropQ)) $tv)))
        ;     ((: (rule-proof conjunction fact-P fact-Q)
        ;     (And (PropP) (PropQ))
        ;     (STV 1.0 1.0))))


            ;; Test for P ^ Q ^ S ^ T -> R
            ; facts
        ; !(compileadd kb (: fact-P (PropP) (STV 1.0 1.0)))
        ; !(compileadd kb (: fact-Q (PropQ) (STV 1.0 1.0)))
        ; !(compileadd kb (: fact-S (PropS) (STV 1.0 1.0)))
        ; !(compileadd kb (: fact-T (PropT) (STV 1.0 1.0)))

            ;     !(compileadd kb
                ;         (: PQST-impl-R
                    ;             (Implication
                    ;             (Premises (PropP) (PropQ) (PropS) (PropT))
                    ;             (Conclusions (PropR)))
                    ;             (STV 1.0 1.0)))

                        ;                 !(test
                        ;                 (collapse(query 300 kb (: $prf (PropR) $tv)))
                            ;                     ((: (rule-proof PQST-impl-R
                                ;                         (rule-proof conjunction
                                ;                         fact-P fact-Q fact-S fact-T))
                            ;                     (PropR)
                            ;                     (STV 1.0 1.0))))
                            
                            
                                ;;  Test for P∧Q -> R

                            ; !(compileadd kb (: fact-P (PropP) (STV 1.0 1.0)))
                            ; !(compileadd kb (: fact-Q (PropQ) (STV 1.0 1.0)))

                                ;     !(compileadd kb
                                    ;         (: PQ-impl-R
                                        ;             (Implication
                                        ;             (Premises (And (PropP) (PropQ)))
                                        ;             (Conclusions (PropR)))
                                        ;             (STV 1.0 1.0)))

                                            ;                 !(test
                                            ;                 (collapse(query 300 kb (: $prf (PropR) $tv)))
                                                ;                     ((: (rule-proof PQ-impl-R
                                                ;                     (conjunction fact-P fact-Q))
                                                ;                     (PropR)
                                                ;                     (STV 1.0 1.0))))


                                                    ;; Test for P -> Q,Q -> R Therefore P -> R through chaining
                                
                                                ; !(compileadd kb (: fact-P (PropP) (STV 1.0 1.0)))

                                                    ;     !(compileadd kb
                                                        ;         (: P-impl-Q
                                                            ;             (Implication
                                                            ;             (Premises (PropP))
                                                            ;             (Conclusions (PropQ)))
                                                            ;             (STV 1.0 1.0)))

                                                                ;                 !(compileadd kb
                                                                    ;                     (: Q-impl-R
                                                                        ;                         (Implication
                                                                        ;                         (Premises (PropQ))
                                                                        ;                         (Conclusions (PropR)))
                                                                        ;                         (STV 1.0 1.0)))

                                                                            ;                             !(test
                                                                            ;                             (collapse(query 300 kb (: $prf (PropR) $tv)))
                                                                                ;                                 ((: (rule-proof
                                                                                    ;                                     Q-impl-R
                                                                                ;                                 (rule-proof P-impl-Q fact-P))
                                                                                ;                                 (PropR)
                                                                                ;                                 (STV 1.0 1.0))))
                                                                                    ;     ;;  Test for P ^ Q -> R, but only P exists

                                                                                ; !(compileadd kb (: fact-P (PropP) (STV 1.0 1.0)))

                                                                                    ;     !(compileadd kb
                                                                                        ;         (: PQ-impl-R
                                                                                            ;             (Implication
                                                                                            ;             (Premises (PropP) (PropQ))
                                                                                            ;             (Conclusions (PropR)))
                                                                                            ;             (STV 1.0 1.0)))

                                                                                                ;                 !(test
                                                                                                ;                 (collapse(query 100 kb (: $prf (PropR) $tv)))
                                                                                                ;                 ())


                                                                                                    ;; P -> R, Q -> R produces merged proof of R

                                                                                                !(compileadd kb (: fact-P (PropP) (STV 1.0 1.0)))
                                                                                                !(compileadd kb (: fact-Q (PropQ) (STV 1.0 1.0)))

                                                                                                    !(compileadd kb
                                                                                                        (: P-impl-R
                                                                                                            (Implication
                                                                                                            (Premises (PropP))
                                                                                                            (Conclusions (PropR)))
                                                                                                            (STV 1.0 1.0)))

                                                                                                                !(compileadd kb
                                                                                                                    (: Q-impl-R
                                                                                                                        (Implication
                                                                                                                        (Premises (PropQ))
                                                                                                                        (Conclusions (PropR)))
                                                                                                                        (STV 1.0 1.0)))

                                                                                                                            !(test
                                                                                                                            (collapse(query 100 kb (: $prf (PropR) $tv)))
                                                                                                                                ((: (merge/revision
                                                                                                                                (rule-proof Q-impl-R fact-Q)
                                                                                                                                (rule-proof P-impl-R fact-P))
                                                                                                                                (PropR)
                                                                                                                                (STV 1.0 0.999950002499875))))

                                                                    

        

 
