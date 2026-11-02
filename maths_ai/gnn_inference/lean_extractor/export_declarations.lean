import Lean
open Lean

def main (args : List String) : IO Unit := do
  let modules : Array Import :=
    args.toArray.map fun name => { module := name.toName }
  let env ← importModules modules {}
  let stdout ← IO.getStdout

  stdout.putStr "["
  let mut first := true
  for entry in env.constants.toList do
    if first then
      first := false
    else
      stdout.putStr ","
    stdout.putStr (toJson entry.1.toString).compress

  stdout.putStr "]\n"
  stdout.flush
