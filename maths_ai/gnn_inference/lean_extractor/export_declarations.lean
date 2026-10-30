import Lean
open Lean

def main (args : List String) : IO Unit := do
  let modules : Array Import := args.toArray.map fun name => { module := name.toName }
  let env ← importModules modules {}
  let names := env.constants.toList.map fun entry => entry.1.toString
  IO.println (toJson names).compress
