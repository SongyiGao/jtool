from abc import ABC, abstractmethod


class Distribution(ABC):
    
    @abstractmethod
    def log_prob(self, x):
        pass 
    
    @abstractmethod
    def sample(self, x):
        pass 
    
    @abstractmethod
    def entropy(self):
        pass