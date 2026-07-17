from typing import Dict, Type, Any, Callable

class Container:
    """Dependency Injection Container for registration and resolution of platform services."""
    
    _bindings: Dict[Type[Any], Callable[[], Any]] = {}
    _instances: Dict[Type[Any], Any] = {}

    @classmethod
    def register(cls, interface: Type[Any], factory: Callable[[], Any], singleton: bool = True) -> None:
        """Register a factory function for a given interface type."""
        cls._bindings[interface] = factory
        if not singleton:
            # If not a singleton, remove any existing cached instance
            cls._instances.pop(interface, None)

    @classmethod
    def resolve(cls, interface: Type[Any]) -> Any:
        """Resolve an instance for the requested interface type."""
        if interface in cls._instances:
            return cls._instances[interface]
            
        if interface in cls._bindings:
            factory = cls._bindings[interface]
            instance = factory()
            cls._instances[interface] = instance
            return instance
            
        raise KeyError(f"Interface '{interface.__name__}' is not registered in the DI Container.")

    @classmethod
    def clear(cls) -> None:
        """Clear all bindings and cached singleton instances."""
        cls._bindings.clear()
        cls._instances.clear()
