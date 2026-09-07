# GSD context snapshot (2026-08-25T10:31:29.607Z)

## Top project memories
- [MEM009] (convention) Each local issue must be implemented on its own dedicated branch and end in its own open GitHub PR; moving its tracker file to `.gsd/issues/done/` is not sufficient completion evidence. Request explicit user confirmation immediately before the outward-facing push and PR creation.
- [MEM001] (convention) Configuration is supplied as untyped JSON/YAML fragments, shallow-merged in supplied order before cross-configuration and validation. Keep configuration values defined once in the merged object rather than duplicating defaults in callers.
- [MEM002] (pattern) CLI entry points run inside `version_controlled_execution_context`, which owns output-directory creation, provenance capture, and random seeding. New run-producing entry points should use the same context rather than recreate this lifecycle.
- [MEM003] (convention) Tests create complete configurations through shared fixtures in `test/conftest.py` and fixed test config files; tests should not mutate configuration values at runtime to create scenarios.
- [MEM008] (convention) Local issue tracker files live in `.gsd/issues/open/`; completed issues are moved to `.gsd/issues/done/` in the same change for traceability.
- [MEM004] (architecture) Configuration composition Chose: Accept untyped JSON or YAML configuration fragments, expand directories recursively, then shallow-merge them in caller order before cross-configuration and validation.. Rationale: This makes a run configurable through composable domain-specific fragments while preserving one merged configuration as the source of truth..

## Recent gsd_exec runs
- [1a50efb9-941f-4ec5-8025-d0d2a945db81] bash exit:0 — find unified loss assembly
- [88558bc6-3fc8-4ba1-8a30-3ac8a73a9e56] bash exit:0 — inspect current unified loss and tests for migration
- [11205c36-b276-42bf-85a4-9cfb39fc82f9] bash exit:0 — show unified loss definition
- [8c8fa86e-98c2-4d84-a137-c842409e806c] bash exit:0 — inspe
…[truncated]
