"""
Pytest configuration and shared fixtures for all tests.
This file is automatically discovered by pytest.
"""
from data_tools.data_generation import DataGeneration
from data_tools.detector.detector_effect import DetectorEffect
from frame.command_line.handle_args import create_config_from_paths
from frame.context.execution_context import version_controlled_execution_context
from test.environment import DEFAULT_CONFIG_PATHS, wrap_with_command_line_args

from pytest import fixture

from argparse import Namespace


@fixture(scope="session", autouse=True)
def session_execution_context():
    args = Namespace(
        debug=True,
        no_build=True,
        out_dir="results",
        only_train=False,
    )
    config = create_config_from_paths(
        config_paths=list(DEFAULT_CONFIG_PATHS.values()),
        is_plot=True,
        out_dir=args.out_dir,
        plot_in_place=True,
    )
    with version_controlled_execution_context(
        config=config,
        config_paths=list(DEFAULT_CONFIG_PATHS.values()),
        command_line_args=wrap_with_command_line_args(DEFAULT_CONFIG_PATHS),
        args=args,
    ) as context:
        yield context


@fixture(scope="function")
def function_execution_context(
    request,
    session_execution_context,
):
    config_paths = DEFAULT_CONFIG_PATHS.copy()
    if request.param:
        config_paths.update(request.param)

    args = Namespace(
        debug=session_execution_context.is_debug_mode,
        no_build=session_execution_context.is_no_build,
        out_dir=session_execution_context.unique_out_dir,
        only_train=session_execution_context.is_only_train,
    )
    config = create_config_from_paths(
        config_paths=list(config_paths.values()),
        is_plot=True,
        out_dir=session_execution_context.unique_out_dir / request.node.name,
        plot_in_place=True,
    )
    with version_controlled_execution_context(
        config=config,
        config_paths=list(config_paths.values()),
        command_line_args=wrap_with_command_line_args(config_paths),
        args=args,
    ) as context:
        yield context


@fixture(scope="function")
def data_generation(request, function_execution_context):
    return DataGeneration(function_execution_context)


@fixture(scope="function")
def detector_effect(request, function_execution_context):
    return DetectorEffect(function_execution_context)


def pytest_runtest_setup(item):
    """Remove timeout for tests marked with 'long'."""
    if "long" in item.keywords:
        # Disable timeout for tests marked as 'long'
        item.timeout = 60 * 15
