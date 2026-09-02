!(import! &self (library lib_import))

;; PeTTaChainer is vendored locally in the Docker image at /opt/pettachainer.
;; No git-import! — avoids runtime network access and pinned-commit drift.
!(import! &self (library PeTTaChainer "/opt/pettachainer/petta/petta_chainer"))