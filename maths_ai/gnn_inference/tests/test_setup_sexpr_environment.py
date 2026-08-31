from __future__ import annotations

import unittest
from unittest.mock import patch

from maths_ai.gnn_inference.scripts.setup_sexpr_environment import (
    LEAN_TOOLCHAIN,
    _ensure_toolchain,
)


class SetupSExprEnvironmentTests(unittest.TestCase):
    @patch("maths_ai.gnn_inference.scripts.setup_sexpr_environment._run")
    @patch("maths_ai.gnn_inference.scripts.setup_sexpr_environment._output")
    def test_already_installed_toolchain_is_not_reinstalled(self, output, run):
        output.return_value = f"{LEAN_TOOLCHAIN}\nleanprover/lean4:v4.29.1 (default)"
        _ensure_toolchain(LEAN_TOOLCHAIN)
        output.assert_called_once_with("elan", "toolchain", "list")
        run.assert_not_called()

    @patch("maths_ai.gnn_inference.scripts.setup_sexpr_environment._run")
    @patch("maths_ai.gnn_inference.scripts.setup_sexpr_environment._output")
    def test_missing_toolchain_is_installed(self, output, run):
        output.return_value = "leanprover/lean4:v4.29.1"
        _ensure_toolchain(LEAN_TOOLCHAIN)
        run.assert_called_once_with("elan", "toolchain", "install", LEAN_TOOLCHAIN)


if __name__ == "__main__":
    unittest.main()
