from dataclasses import dataclass


SAMPLE_A_NAME = "A"
SAMPLE_B_NAME = "B"


@dataclass(frozen=True)
class TrainingNames:
    numerator: str
    denominator: str


def training_name(sample_name: str, is_numerator: bool) -> str:
    kind = "numerator" if is_numerator else "denominator"
    return f"{sample_name}_{kind}"


def training_names_for_sample(sample_name: str) -> TrainingNames:
    return TrainingNames(
        numerator=training_name(sample_name, is_numerator=True),
        denominator=training_name(sample_name, is_numerator=False),
    )


SAMPLE_A_TRAINING_NAMES = training_names_for_sample(SAMPLE_A_NAME)
SAMPLE_B_TRAINING_NAMES = training_names_for_sample(SAMPLE_B_NAME)


def symmetric_training_names(train_for_nuisances: bool) -> list[str]:
    if train_for_nuisances:
        return [
            SAMPLE_A_TRAINING_NAMES.numerator,
            SAMPLE_A_TRAINING_NAMES.denominator,
            SAMPLE_B_TRAINING_NAMES.numerator,
            SAMPLE_B_TRAINING_NAMES.denominator,
        ]
    return [
        SAMPLE_A_TRAINING_NAMES.numerator,
        SAMPLE_B_TRAINING_NAMES.numerator,
    ]
