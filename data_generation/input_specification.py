from abc import ABC,abstractmethod

class InputSpecification(ABC):
    def get_params():
        pass
    def get_param_range(param):
        pass        
        
class SOverKInputSepcification(InputSpecification):
    pass

class SKSplitInputSpecification(InputSpecification):
    pass
