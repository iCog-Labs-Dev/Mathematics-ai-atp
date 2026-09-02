
!(import! &self ../axioms/repos/PeTTaChainer/pettachainer/metta/petta_chainer)

;;facts 
!(compileadd kb (: fact-P (PropP) (STV 1.0 1.0)))
!(compileadd kb (: fact-P-impl-Q (Implication (Premises (PropP)) (Conclusions (PropQ))) (STV 1.0 1.0)))

 !(test
    (collapse(query 20 kb (: $prf (PropQ) $tv)))
        ((: (rule-proof fact-P-impl-Q fact-P) (PropQ) (STV 1.0 1.0))))