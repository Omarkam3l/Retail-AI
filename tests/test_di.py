import pytest
from src.common.di import Container
from src.common.interfaces import Lifecycle

class MockService(Lifecycle):
    def initialize(self) -> None:
        self.initialized = True
        
    def shutdown(self) -> None:
        self.initialized = False

def test_container_singleton_registration_and_resolution():
    Container.clear()
    
    # Register a factory
    Container.register(MockService, lambda: MockService())
    
    # Resolve service
    service1 = Container.resolve(MockService)
    service2 = Container.resolve(MockService)
    
    assert isinstance(service1, MockService)
    # Check that it resolves to the same instance (singleton by default)
    assert service1 is service2

def test_container_not_registered_raises_key_error():
    Container.clear()
    
    class UnregisteredService:
        pass
        
    with pytest.raises(KeyError) as exc_info:
        Container.resolve(UnregisteredService)
        
    assert "UnregisteredService" in str(exc_info.value)
