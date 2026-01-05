import pytest


def test_illegal_data_request(
        data_generation,
):
    with pytest.raises(KeyError):
        _, _ = data_generation["non_existing_dataset"]


def test_resampling_exhaustion(
        data_generation,
):
    _, _ = data_generation["A"]

    # Draws more data than there is left, due to resampling without replacement
    with pytest.raises(ValueError):
        _, _ = data_generation["A"]