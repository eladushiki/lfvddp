# LFVDDP

Implementing ML machinery to differentiate between similar and different pairs of physically equivalent datasets.

## Development Principles
- THE ONE DEFINITION RULE: Every variable value should be implemted once. Any logic should be implemented once and reused. Have a single source of truth.
- SINGLE PURPOSE RULE: Each function, class or file should have a single purpose. If there are many, split.
- TESTING PHILOSOPHY: Any complex enough (not semantic) logic in the codebase should be tested at least once. Any configuration should be tested at least once with all its options, but not all possible combinations of them - just the interesting ones and the edge cases. Testing logic should never be tested, its validation comes from succeeding to run the project. No testing needed for plots!