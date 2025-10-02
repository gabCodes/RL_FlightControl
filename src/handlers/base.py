from abc import ABC, abstractmethod

class TaskHandler(ABC):
    """Base class for task handlers."""
    
    @abstractmethod    
    def give_initial_state(self, output):
        """Return the initial state for the training."""
        raise NotImplementedError
    
    @abstractmethod
    def random_reference(self, timestep):
        """Return the randomly generated reference for the current timestep."""
        raise NotImplementedError
    
    @abstractmethod
    def eval_reference(self, timestep):
        """Return the fixed eval reference for the current timestep."""
        raise NotImplementedError
    
    @abstractmethod
    def sample_action(self, state):
        """Return the sampled action."""
        raise NotImplementedError
    
    @abstractmethod
    def mean_action(self, state):
        """Return the mean action."""
        raise NotImplementedError
    
    @abstractmethod
    def state_list(self, output):
        """Return the list of states."""
        raise NotImplementedError
    
    @abstractmethod
    def action_list(self,action):
        """Return the list of actions."""
        raise NotImplementedError
    
    @abstractmethod
    def ref_list(self, reference):
        """Return the list of references."""
        raise NotImplementedError
    
    @abstractmethod
    def error_list(self, output, reference):
        """Return the reference error"""
        raise NotImplementedError
    
    @abstractmethod
    def add_buffer(self, state, action, next_state, reward, done):
        """Add experience to the replay buffer."""
        raise NotImplementedError
    
    @abstractmethod
    def compute_state_and_reward(self, reference):
        """Return the next state and reward."""
        raise NotImplementedError