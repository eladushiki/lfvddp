from types import ModuleType
from typing import Any, Callable


def retrieve_from_module(
        module: ModuleType,
        default_value: Any = None,
):

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        
        def wrapped_func(*args, **kwargs):

            parameter_key = func(*args, **kwargs)

            return _retrieve_from_module(
                module=module,
                parameter_key=parameter_key,
                default_value=default_value,
            )
        
        return wrapped_func
    return decorator


def _retrieve_from_module(
        module: ModuleType,
        parameter_key: str,
        default_value: Any = None,
) -> Any:
    """
    Retrieve a value from a module using a string key.
    
    Args:
        module (ModuleType): The module to retrieve the value from.
        parameter_key (str): The key to retrieve the value.
        default_value (Any, optional): The default value to return if the key is not found. Defaults to None.
    
    Returns:
        Any: The retrieved value or the default value.
    """
    if not parameter_key:
        return default_value
    
    try:
        return getattr(module, parameter_key)
    except AttributeError:
        raise ValueError(f"Module '{module.__name__}' does not have attribute '{parameter_key}'")
