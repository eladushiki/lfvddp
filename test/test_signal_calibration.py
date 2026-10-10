import json
from pathlib import Path

import pytest

from data_tools.detector.analytic_efficiency import generated_detector_efficiency
from data_tools.event_generation.background import ExponentialBackground
from data_tools.profile_likelihood import (
    calc_injected_t_significance_by_sqrt_q0_continuous,
)
from data_tools.signal_calibration import (
    calc_n_signal_events_for_config,
    calc_n_signal_events_for_generated_signal,
    calc_n_signal_events_for_target_injected_t_significance,
    main,
)
from data_tools.signal_calibration_config import load_signal_calibration_config


@pytest.mark.parametrize(
    "dataset,detector",
    [
        (
            "test/configs/dataset/calibration_signal_a_1D.json",
            "test/configs/detector/analytic_efficiency_1D.json",
        ),
        (
            "test/configs/dataset/calibration_signal_b_1D.json",
            "test/configs/detector/analytic_efficiency_1D.json",
        ),
        (
            "configs/basic-generated/generated_dataset_config.json",
            "test/configs/detector/analytic_efficiency_2D.json",
        ),
        (
            "test/configs/dataset/calibration_signal_a_1D.json",
            "test/configs/detector/basic_1D_detector_config.json",
        ),
    ],
)
def test_config_calibration_matches_detector_level_forward_calculation(
    dataset, detector
):
    config = load_signal_calibration_config([Path(dataset), Path(detector)])
    parameters = config.signal_dataset_parameters
    count = calc_n_signal_events_for_config(config, 1.6)
    arguments = {
        "background_pdf": parameters.dataset_generated__background_pdf,
        "signal_pdf": parameters.dataset_generated__signal_pdf,
        "n_background_events": parameters.dataset__mean_number_of_background_events,
        "n_signal_events": count,
        "upper_limit": parameters.dataset_generated__integration_upper_limits,
    }
    forward = calc_injected_t_significance_by_sqrt_q0_continuous(
        **arguments,
        detector_efficiency=generated_detector_efficiency(config, parameters),
    )
    assert forward == pytest.approx(1.6, abs=1e-6)
    generated = calc_injected_t_significance_by_sqrt_q0_continuous(**arguments)
    if "analytic_efficiency" in detector:
        assert generated > forward
    else:
        assert generated == pytest.approx(forward)


def test_explicit_generator_api_passes_efficiency_to_the_inverse():
    specification = {"function": "exponential_background"}
    count = calc_n_signal_events_for_generated_signal(
        specification,
        {"function": "nonlocal_signal"},
        1,
        25000,
        1.6,
        detector_efficiency=lambda x: 0.9,
    )
    from data_tools.event_generation.signal import NonlocalSignal

    forward = calc_injected_t_significance_by_sqrt_q0_continuous(
        ExponentialBackground(1).pdf,
        NonlocalSignal(1).pdf,
        25000,
        count,
        upper_limit=NonlocalSignal(1).integration_upper_limits,
        detector_efficiency=lambda x: 0.9,
    )
    assert forward == pytest.approx(1.6, abs=1e-6)


def test_zero_acceptance_cannot_reach_positive_target():
    pdf = ExponentialBackground(1).pdf
    with pytest.raises(ValueError, match="max_n_signal_events"):
        calc_n_signal_events_for_target_injected_t_significance(
            pdf,
            pdf,
            100,
            1.6,
            upper_limit=5,
            max_n_signal_events=16,
            detector_efficiency=lambda x: 0,
        )


def test_config_cli_calibrates_five_targets_with_efficiency(capsys):
    paths = [
        Path("test/configs/dataset/calibration_signal_a_1D.json"),
        Path("test/configs/detector/analytic_efficiency_1D.json"),
    ]
    targets = [1.6, 3.2, 4.8, 6.4, 8.0]
    main(["--configs", *map(str, paths), "--target-significance", *map(str, targets)])
    results = json.loads(capsys.readouterr().out)
    assert [r["target_significance"] for r in results] == targets
    config = load_signal_calibration_config(paths)
    parameters = config.signal_dataset_parameters
    for result, target in zip(results, targets):
        forward = calc_injected_t_significance_by_sqrt_q0_continuous(
            parameters.dataset_generated__background_pdf,
            parameters.dataset_generated__signal_pdf,
            parameters.dataset__mean_number_of_background_events,
            result["mean_signal_events"],
            upper_limit=parameters.dataset_generated__integration_upper_limits,
            detector_efficiency=generated_detector_efficiency(config, parameters),
        )
        assert forward == pytest.approx(target, abs=1e-6)


def test_config_calibration_rejects_loaded_signal():
    config = load_signal_calibration_config(
        [
            Path("configs/basic-loaded/loaded_dataset_config.json"),
            Path("configs/basic-loaded/detector_config.json"),
        ]
    )
    with pytest.raises(TypeError, match="generated dataset"):
        calc_n_signal_events_for_config(config, 1.6)


@pytest.mark.parametrize("extra", [[], ["--number-of-dimensions", "1"]])
def test_cli_rejects_incomplete_or_mixed_input_modes(extra):
    args = (
        ["--background-generator", '{"function":"exponential_background"}']
        if not extra
        else ["--configs", "test/configs/dataset/calibration_signal_a_1D.json", *extra]
    )
    with pytest.raises(SystemExit):
        main([*args, "--target-significance", "1.6"])
