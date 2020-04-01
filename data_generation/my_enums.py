from enum import Enum
from pricer import BSFormulaPricer
from juzhong_pricer import JuZhongPricer
from whaley_pricer import WhaleyPricer
from juzhong_whaley_pricer import JuZhongWhaleyPricer
from generator import UniformGenerator,HaltonGenerator


class Pricers(Enum):
    BSFormula=1
    JuZhong= 2
    Whaley=3
    JuZhongWhaley=4
    
    @staticmethod
    def from_enum(e):
        if e == Pricers.BSFormula:
            return BSFormulaPricer()
        if e == Pricers.JuZhong:
            return JuZhongPricer()
        if e == Pricers.Whaley:
            return WhaleyPricer()
        if e == Pricers.JuZhongWhaley:
            return JuZhongWhaleyPricer()
        raise ValueError("Invalid Pricer Enum {}".format(e))
        
    @staticmethod
    def american_pricers():
        return [Pricers.JuZhong,Pricers.Whaley,Pricers.JuZhongWhaley]
    
    @staticmethod
    def european_pricers():
        return [Pricers.BSFormula]
        
class Generators(Enum):
    Uniform = 1
    Halton = 2
    
    @staticmethod
    def from_enum(e):
        if e == Generators.Uniform:
            return UniformGenerator()
        if e == Generators.Halton:
            return HaltonGenerator()