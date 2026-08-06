from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Dict, List, Mapping, Sequence, Union

from camel_converter import to_pascal
from data_tools.data_utils import DataSet
from data_tools.event_generation.types import FLOAT_OR_ARRAY
from frame.module_retriever import _retrieve_from_module
import numpy as np


@dataclass(frozen=True)
class GeneratorSpec:
    """Configuration for one dataset distribution."""

    function: str
    arguments: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, value: Mapping[str, Any]) -> "GeneratorSpec":
        if not isinstance(value, Mapping):
            raise TypeError(
                "A generator specification must be an object containing "
                "'function' and optional 'arguments' fields."
            )

        unexpected_fields = set(value) - {"function", "arguments"}
        if unexpected_fields:
            raise ValueError(
                "Unexpected generator specification fields: "
                f"{sorted(unexpected_fields)}"
            )

        function = value.get("function")
        if not isinstance(function, str) or not function:
            raise ValueError(
                "Generator specification field 'function' must be a non-empty string."
            )

        arguments = value.get("arguments", {})
        if not isinstance(arguments, Mapping):
            raise TypeError(
                "Generator specification field 'arguments' must be an object."
            )

        return cls(function=function, arguments=dict(arguments))

    def as_dict(self) -> Dict[str, Any]:
        return {"function": self.function, "arguments": self.arguments}


GeneratorSelection = Union[GeneratorSpec, List[GeneratorSpec]]
GeneratorSelectionConfig = Union[Mapping[str, Any], Sequence[Mapping[str, Any]]]


def normalize_generator_selection(
    selection: GeneratorSelectionConfig,
) -> GeneratorSelection:
    """Parse a joint generator object or an independent-generator list."""
    if isinstance(selection, Mapping):
        return GeneratorSpec.from_config(selection)

    if isinstance(selection, Sequence) and not isinstance(selection, (str, bytes)):
        if not selection:
            raise ValueError("Generator specification lists cannot be empty.")
        return [GeneratorSpec.from_config(spec) for spec in selection]

    raise TypeError(
        "A generator must be configured as one specification object or a list "
        "of specification objects."
    )


def generator_selection_as_config(
    selection: GeneratorSelection,
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    if isinstance(selection, GeneratorSpec):
        return selection.as_dict()
    return [spec.as_dict() for spec in selection]


def describe_generator_selection(selection: GeneratorSelection) -> str:
    def describe_spec(spec: GeneratorSpec) -> str:
        arguments = ", ".join(
            f"{name}={value}" for name, value in sorted(spec.arguments.items())
        )
        return f"{spec.function}({arguments})" if arguments else spec.function

    if isinstance(selection, GeneratorSpec):
        return describe_spec(selection)
    return "independent[" + ", ".join(describe_spec(spec) for spec in selection) + "]"


def validate_generated_dataset(
    dataset: DataSet,
    amount: int,
    number_of_dimensions: int,
    generator_name: str,
) -> None:
    if not isinstance(dataset, DataSet):
        raise TypeError(
            f"Generator '{generator_name}' must return a DataSet, got "
            f"{type(dataset).__name__}."
        )
    if dataset.n_samples != amount or dataset.n_observables != number_of_dimensions:
        raise ValueError(
            f"Generator '{generator_name}' returned shape "
            f"({dataset.n_samples}, {dataset.n_observables}); expected "
            f"({amount}, {number_of_dimensions})."
        )


class DataDistribution(ABC):
    """
    A distribution with certain properties.

    When implementing a new distribution, the responsibility
    is on the developer to ensure the correctness and the
    normalization of the implemented functions.
    """

    def __init__(
        self,
        number_of_dimensions: int,
        domain_min: float = 0,
        domain_max: float = 1e2,
        domain_granularity: int = 100000,
    ):
        if number_of_dimensions <= 0:
            raise ValueError("A distribution must have at least one dimension.")
        if domain_min >= domain_max:
            raise ValueError("domain_min must be smaller than domain_max.")
        if domain_granularity <= 0:
            raise ValueError("domain_granularity must be positive.")

        self._number_of_dimensions = number_of_dimensions
        self._domain_min = domain_min
        self._domain_max = domain_max
        self._domain_granularity = domain_granularity

    def generate_amount(
        self,
        amount: int,
    ) -> DataSet:
        """
        Generate a sample of the distribution.

        This implementation draws from the pdf can be used in
        inherited classes, but need to be overriden explicitly.
        """
        rng = np.linspace(
            self._domain_min,
            self._domain_max,
            self._domain_granularity,
        )

        probabilities = np.array([self.pdf(x) for x in rng])
        probabilities /= probabilities.sum()

        return DataSet(np.random.choice(
            rng,
            size=(amount, self._number_of_dimensions),
            replace=True,
            p=probabilities,
        ))

    @abstractmethod
    def pdf(self, x: FLOAT_OR_ARRAY) -> FLOAT_OR_ARRAY:
        """
        Probability density function of the distribution.

        Should sum to 1.
        """
        pass


class IndependentDimensionsDistribution(DataDistribution):
    """Compose separately configured 1D distributions into one dataset."""

    def __init__(self, distributions: List[DataDistribution], names: List[str]):
        if not distributions:
            raise ValueError("Independent distribution composition cannot be empty.")
        if len(distributions) != len(names):
            raise ValueError("Every independent distribution must have a name.")
        super().__init__(len(distributions))
        self._distributions = distributions
        self._names = names

    def generate_amount(self, amount: int) -> DataSet:
        generated_dimensions = []
        for distribution, name in zip(self._distributions, self._names):
            generated = distribution.generate_amount(amount)
            validate_generated_dataset(generated, amount, 1, name)
            generated_dimensions.append(generated.events[:, 0])

        return DataSet(np.column_stack(generated_dimensions))

    def pdf(self, x: FLOAT_OR_ARRAY) -> FLOAT_OR_ARRAY:
        values = np.asarray(x)
        if values.ndim == 0:
            if self._number_of_dimensions != 1:
                raise ValueError(
                    "An N-D independent PDF must be evaluated with one coordinate "
                    "per dimension."
                )
            return self._distributions[0].pdf(x)
        if values.shape[-1] != self._number_of_dimensions:
            raise ValueError(
                f"Expected {self._number_of_dimensions} coordinates, got "
                f"{values.shape[-1]}."
            )

        marginal_densities = [
            distribution.pdf(values[..., dimension])
            for dimension, distribution in enumerate(self._distributions)
        ]
        return np.prod(np.asarray(marginal_densities), axis=0)


def resolve_generator(
    module: ModuleType,
    selection: GeneratorSelection,
    number_of_dimensions: int,
) -> DataDistribution:
    """Resolve a joint N-D generator or independent 1D generator specs."""

    def instantiate(spec: GeneratorSpec, dimensions: int) -> DataDistribution:
        class_name = to_pascal(spec.function)
        distribution_class = _retrieve_from_module(module, class_name)
        if not isinstance(distribution_class, type) or not issubclass(
            distribution_class, DataDistribution
        ):
            raise TypeError(
                f"Configured generator '{spec.function}' must name a "
                "DataDistribution subclass."
            )
        return distribution_class(dimensions, **spec.arguments)

    if isinstance(selection, GeneratorSpec):
        return instantiate(selection, number_of_dimensions)

    if len(selection) not in (1, number_of_dimensions):
        raise ValueError(
            "An independent generator list must contain either one generator to "
            f"repeat or exactly {number_of_dimensions} generators; got "
            f"{len(selection)}."
        )

    dimension_specs = selection * number_of_dimensions if len(selection) == 1 else selection
    return IndependentDimensionsDistribution(
        distributions=[instantiate(spec, 1) for spec in dimension_specs],
        names=[spec.function for spec in dimension_specs],
    )
