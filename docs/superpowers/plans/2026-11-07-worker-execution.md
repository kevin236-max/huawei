# Worker execution and soft deadline implementation plan

**Goal:** Main defines ideas and tuning strategies; workers execute them and receive a single wrap-up instruction at the soft deadline.

**Scope:** Approved conversation design. Keep three rounds, three worker slots, existing candidate interface and host-owned scores. Preserve existing workspace changes.

**Implementation:**
- [x] Add runtime tests using real subprocess pipes: one queued steering message, completion after the soft deadline, normal early completion and hard timeout.
- [x] Add handoff tests: preserve a report without a file, preserve an unscored draft, and prefer existing evaluated artifacts when available.
- [x] Update main/subagent prompts with bounded ideas, main-owned tuning strategies and two representative examples (zero references / two references).
- [x] Add a wrap-up prompt; use 480 seconds as soft deadline and 600 seconds additional finishing allowance, bounded by the global deadline.
- [x] Pass the hard deadline to gateway/evaluator/session, and the soft deadline only to RPC steering. Record the injected prompt and preserve the existing process cleanup.
- [x] Update configuration validation, integration coverage and timeout documentation.
- [x] Run the relevant unittest suites and review the changes against the approved design. No paid model calls or experiment reruns.

**Validation:** `python -B -m unittest discover -s tests -v` using the existing huawei Python environment. Runtime tests use a lightweight RPC subprocess; integration tests retain real candidate fitting/scoring with local model and sandbox stand-ins.

**Result:** 57 tests run, 54 passed, 3 socket-dependent tests skipped by the restricted environment. Prompt templates and JSON examples validated. Snapshot-scoped code review found no actionable issues. Real-model timeout rates remain to be measured in a later experiment.
