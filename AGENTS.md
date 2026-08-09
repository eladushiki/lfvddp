# LFVDDP

Implementing ML machinery to differentiate between similar and different pairs of physically equivalent datasets.

## Development Principles
- THE ONE DEFINITION RULE: Every variable value should be implemted once. Any logic should be implemented once and reused. Have a single source of truth.
- SINGLE PURPOSE RULE: Each function, class or file should have a single purpose. If there are many, split.
- TESTING PHILOSOPHY:
  - Any complex enough (not semantic) logic in the codebase should be tested at least once. Simple additions should not. Any configuration should be tested at least once with all its options, but not all possible combinations of them - just the interesting ones and the edge cases.
  - Testing logic should never be tested, its validation comes from succeeding to run the project.
  - NO TESTING NEEDED FOR PLOT OR PLOT UTILITIES!
  When adding a test, examine `conftest.py`, `test_train.py:test_learning` and it's use of fixtures. Mimic config file creation and fixture use rather than adding filename variables and/or editing configuration values in runtime.
- DOCUMENTATION: Change README.md indicating any significant user facing change.
- DIMENSIONAL PLOT PARITY: When changing a 1D graph or its n-dimensional equivalent, locate the corresponding graph function and apply the equivalent change where appropriate, or explicitly verify that no corresponding change is needed.
