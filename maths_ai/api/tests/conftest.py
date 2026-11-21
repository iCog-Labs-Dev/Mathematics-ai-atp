def pytest_configure(config):
    # --run-pantograph-live is registered by the inference tests' conftest; the
    # live service test runs whenever its paths are set and reads the flag
    # only to fail instead of skipping in a required run.
    config.addinivalue_line("markers", "pantograph_live: actual custom Lean executable integration")
