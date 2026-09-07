# Issue 011: Nuisance Configuration in Train Config

**Status:** Open

## Statement
Move the detector bin bound and counting configuration parameters to the training config (they are now in the detector config). Have in mind that if the detector nuisances are set to be bins they need said paramters, but if they're set to be a nn then just the number of hidden layer neurons should be given, and not bin bounds and count.

## Acceptance criteria
- Either mixed combination of parameters should result in an error when creating the Config.
- Existing, git backed configuration packs should preserve functionality but update structure.

## Completed
- Moved scalar nuisance binning bounds and counts from detector configuration to `TrainConfig` and made them mutually exclusive with neural nuisance configuration.
- Migrated committed configuration packs and test fixtures; added configuration-creation tests for both invalid mixed combinations.
