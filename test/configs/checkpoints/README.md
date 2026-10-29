# Checkpoint metadata fixtures

`legacy_neural_widths.json` represents metadata saved before neural endpoint widths
were removed from configuration. Its `input_dimension` and `output_dimension`
fields intentionally preserve that old format: the compatibility test checks
that scalar output hints remain readable and a different saved input width is
rejected. They are not parameters accepted by current neural configurations.

The remaining fields identify the checkpoint format, model, likelihood role,
backend, signal and nuisance architectures, configuration fingerprint, and
observable normalization used when the checkpoint was saved.
