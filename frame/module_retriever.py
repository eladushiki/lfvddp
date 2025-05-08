from inspect import signature
from types import ModuleType
from typing import Any, Callable


def retrieve_from_module(
        module: ModuleType,
        default_value: Any = None,
):

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        
        def wrapped_func(*args, **kwargs):

            parameter_key = func(*args, **kwargs)

            try:
                retrieved_value = getattr(module, parameter_key)
                return retrieved_value if retrieved_value is not None else default_value
                
            except AttributeError:
                raise ValueError(f"Module '{module.__name__}' does not have attribute '{parameter_key}'")

        return wrapped_func
    return decorator
