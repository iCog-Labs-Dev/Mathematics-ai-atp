"""
Lean/PyPantograph -> PeTTaChainer translator, updated for the NEW parser format.

This version writes ONE .metta file per resulting Lean subgoal, generates a shell
runner script, captures one .log file per subgoal, and can parse/rank those logs.

The generated runner executes each .metta file using:

    petta <absolute-path-to-metta-file>

Default output layout:

    output_dir/
      <test_name>_goal_0.metta
      <test_name>_goal_0.log
      <test_name>_goal_1.metta
      <test_name>_goal_1.log
      generated_manifest.json
      run_all_generated.sh

The manifest records each generated subgoal file and its corresponding log path.
After running the shell script, you can rank results by parsing logs:

    python lean_to_pettachainer_with_runner_and_ranking.py \\
      --rank-manifest output_dir/generated_manifest.json \\
      --ranking-output output_dir/ranking_results.json

Or generate, run manually, then rank.

This translator matches the updated Metamath parser format where formulas are not
wrapped in (Provable ...). Queries are emitted as:

    !(query 10 kb (: $prf FORMULA $tv))

Variable handling in this adjusted version:

- By default, Lean/Pantograph variable names are preserved.
  Example: P, Q, hP-type formulas stay as P, Q instead of becoming v0, v1.
- If you want the older canonical structural behavior, pass:

    --normalize-variables

Ranking fallback:

- If none of the available subgoal logs contains any `(STV strength confidence)`
  result, ranking mode assigns random fallback scores to the subgoals.
- This is useful as an exploration fallback when PeTTaChainer cannot prove any
  generated subgoal.
- The fallback can be configured with:
      --fallback-distribution uniform|beta
      --fallback-low
      --fallback-high
      --fallback-alpha
      --fallback-beta
      --random-seed
- It can be disabled with:
      --disable-random-fallback

--- MODULARIZATION NOTE ---

All functionality has been moved to the `translator_modules` package.
This file is a backward-compatible re-export facade: any existing code doing
`from translator import X` will continue to work.

For direct module imports, use:
    from translator_modules.constants import LEAN_TO_PETTA
    from translator_modules.parser import parse_sexp_string
    from translator_modules.renderer import render_formula
    etc.
"""

from translator_modules import *  # noqa: F401, F403

if __name__ == "__main__":
    main()
