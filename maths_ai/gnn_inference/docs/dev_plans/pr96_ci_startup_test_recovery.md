# PR #96: bounded startup tests and CI diagnostics

## Evidence and scope

The CI run at PR head `8c4482222555269191ca2354d9ddc5c5d60ce929` failed two model-client factory tests, then waited in the cancellation test until the six-hour job limit. The saved log contains no failure traceback because pytest never completed its session. Dependency installation succeeded with the pinned client 0.7.0.

The pinned client source passes the five factory tests and the inference regression suite locally. The original two CI errors are therefore not yet reproduced. This repair must not describe a suspected interpreter or platform difference as a confirmed production defect.

## Test and CI changes

1. Make fake executable scripts use the interpreter running pytest, rather than whichever `python3` is first on the host PATH. Verify this with a conflicting PATH interpreter. Production launch behavior is unchanged.
2. Bound the shared fake-client test runner. When testing cancellation during startup, wait for either the startup event or operation completion and propagate an early startup exception. Always cancel and await owned tasks. Migrate both factory and worker-pool cancellation tests using that pattern.
3. Add regressions for early startup failure and missing startup notification, retaining existing process-reaping checks.
4. Stop CI on its first failed test, write JUnit results, and retain diagnostics on failure. Set an explicit job deadline and enable bounded faulthandler dumps so a hang cannot hide the original failure for six hours.
5. Test against the published PR revision and pinned client. Preserve unrelated local changes. Report which checks pass and whether a fresh GitHub run is still needed to establish the original CI-specific cause.

## Alternatives

- Increase timeouts or skip the factory tests: rejects real lifecycle coverage and does not explain the failure.
- Relax protocol validation: unsupported by the evidence and would weaken production checks.
- Change production startup without a reproduction: risks changing working runtime behavior. Keep this correction in the test harness and workflow unless further evidence establishes a production defect.

This is a test-harness and CI correction, with no new production architecture, client/REPL pin, model format, or artifact migration.

## Confirmed runner dependency correction

Run `44891098225` passes source lineage and exposes the original startup rejection: `The configured stdbuf is not GNU coreutils stdbuf.` Its runner is Ubuntu 26.04, whose default coreutils provider is uutils. The client deliberately requires GNU stdbuf; do not bypass that validation or skip startup tests.

Provision the existing required dependency explicitly: select Ubuntu 26.04, install its `gnu-coreutils` package, and expose `/usr/bin/gnustdbuf` through a job-local `stdbuf` symlink prepended to PATH with `GITHUB_PATH`. Verify its GNU version before dependency builds or tests. This preserves the host's coreutils provider and the client's strict launch contract. Pinning an older runner would hide the missing dependency declaration; replacing system coreutils is unnecessary.

Reproduce rejection with a uutils-style launcher on PATH, then verify selection and actual fake-client startup with GNU stdbuf. Record workflow and lineage-test changes in a new manifest stage after `6a3b5a152b53c4f97ed02ff4080690d1604a0983`; preserve earlier stages. Validate the exact committed snapshot plus the complete new patch, including its workflow.
