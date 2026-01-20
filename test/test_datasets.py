def test_dataset_normalization(
    function_execution_context,
    data_generation,
):
    A, A_params = data_generation["A"]
    B, B_params = data_generation["B"]

    normalized_A, norm_factor_A = A.get_normalized()
    normalized_B, norm_factor_B = B.get_normalized()

    assert np.max(normalized_A.events) == 1
    assert np.max(normalized_B.events) == 1
    assert np.min(normalized_A.events) == 0
    assert np.min(normalized_B.events) == 0

    assert all((normalized_A * norm_factor_A).events == A.events)
