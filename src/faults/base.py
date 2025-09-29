from abc import ABC, abstractmethod

class FaultHandler(ABC):
    """Base class for task handlers."""
    
    @abstractmethod
    def __getattr__(self, name):
        """Delegate attribute access to the wrapped handler."""
        raise NotImplementedError