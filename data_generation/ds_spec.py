from enum import Enum
import numpy as np

class DsSpec:
    def __init__(self,input_cols,output_cols,gen_ranges,
                 get_input_func,get_output_func,desc):
        self._input_cols = input_cols
        self._output_cols = output_cols
        self.gen_ranges = gen_ranges
        self._get_input_func = get_input_func
        self._get_output_func = get_output_func
        self.desc = desc
    
    def get_input_func(self,data_makers):
        return self._get_input_func(data_makers,self)
    
    def get_output_func(self,data_makers):
        return self._get_output_func(data_makers,self)
    
    def num_input_cols(self):
        return len(self.input_cols())
    
    def num_output_cols(self):
        return len(self.output_cols())
    
    def input_cols(self):
        return self._input_cols
    
    def output_cols(self):
        return self._output_cols

class InputFuncs:
    @staticmethod
    def get_input_func_1(data_makers,spec):
        generator,am_pricer,eu_pricer = data_makers
        def gen_input(num):
            return generator.generate(spec.input_cols(),spec.gen_ranges,num)
        return gen_input
    
    @staticmethod
    def get_input_func_2(data_makers,spec):
        generator,am_pricer,eu_pricer = data_makers
        def gen_input(num):
            raw_cols = [col for col in spec.input_cols() if col != 'EU_Price']
            raw_inputs = generator.generate(raw_cols,spec.gen_ranges,num)
            cols_dict = {col:i for i,col in enumerate(spec.input_cols())}
            def inp(col):
                return raw_inputs[:,cols_dict[col]]
            eu_prices = eu_pricer.get_price(
                    inp('S'),inp('K'),inp('q'),
                    inp('r'),inp('sigma'),inp('T'),-1).reshape((-1,1))
            return np.append(raw_inputs,eu_prices,axis=1)
        return gen_input
    
    @staticmethod
    def get_input_func_3(data_makers,spec):
        generator,am_pricer,eu_pricer = data_makers
        def gen_input(num):
            raw_cols = [col for col in spec.input_cols() if col != 'EU_Price']
            raw_inputs = generator.generate(raw_cols,spec.gen_ranges,num)
            cols_dict = {col:i for i,col in enumerate(spec.input_cols())}
            def inp(col):
                return raw_inputs[:,cols_dict[col]]
            eu_prices = eu_pricer.get_price(
                    inp('S'),1,inp('q'),
                    inp('r'),inp('sigma'),inp('T'),-1).reshape((-1,1))
            return np.append(raw_inputs,eu_prices,axis=1)
        return gen_input  

class OutputFuncs:
    @staticmethod
    def get_output_func_1(data_makers,spec):
        generator,am_pricer,eu_pricer = data_makers
        def gen_output(inputs):
            cols_dict = {col:i for i,col in enumerate(spec.input_cols())}
            def inp(col):
                return inputs[:,cols_dict[col]]
            eu_prices = eu_pricer.get_price(
                    inp('S'),inp('K'),inp('q'),
                    inp('r'),inp('sigma'),inp('T'),-1).reshape((-1,1))
            am_prices = am_pricer.get_price(
                    inp('S'),inp('K'),inp('q'),
                    inp('r'),inp('sigma'),inp('T'),-1).reshape((-1,1))
            return am_prices - eu_prices
        return gen_output
 
    @staticmethod
    def get_output_func_2(data_makers,spec):
        generator,am_pricer,eu_pricer = data_makers
        def gen_output(inputs):
            cols_dict = {col:i for i,col in enumerate(spec.input_cols())}
            def inp(col):
                return inputs[:,cols_dict[col]]
            eu_prices = eu_pricer.get_price(
                    inp('S'),1,inp('q'),
                    inp('r'),inp('sigma'),inp('T'),-1).reshape((-1,1))
            am_prices = am_pricer.get_price(
                    inp('S'),1,inp('q'),
                    inp('r'),inp('sigma'),inp('T'),-1).reshape((-1,1))
            return am_prices - eu_prices
        return gen_output
    
    @staticmethod
    def get_output_func_3(data_makers,spec):
        generator,am_pricer,eu_pricer = data_makers
        def gen_output(inputs):
            cols_dict = {col:i for i,col in enumerate(spec.input_cols())}
            def inp(col):
                return inputs[:,cols_dict[col]]
            eu_prices = inp('EU_Price').reshape((-1,1))
            am_prices = am_pricer.get_price(
                    inp('S'),1,inp('q'),
                    inp('r'),inp('sigma'),inp('T'),-1).reshape((-1,1))
            return am_prices - eu_prices
        return gen_output
    
class DsSpecs(Enum):
    #first number indicates get_input_func
    #second number indicates get_output_func
    #third number is an index
    SPEC_1_1_1 = 1    
    SPEC_2_1_1 = 2
    SPEC_1_2_1 = 3
    SPEC_3_3_1 = 4
    
    @staticmethod
    def from_enum(spec_enum):        
        if spec_enum == DsSpecs.SPEC_1_1_1:
            description = "Vanilla input output specs"
            return DsSpec(input_cols=['S','K','r','q','sigma','T'],
                   output_cols=['Premium'],
                   gen_ranges={'S':(80,120),
                               'K':(80,120),
                               'r':(0.01,0.03),
                               'q':(0,0.03),
                               'sigma':(0.05,0.5),
                               'T':(1/12,3)},
                   get_input_func=InputFuncs.get_input_func_1,
                   get_output_func=OutputFuncs.get_output_func_1,
                   desc=description)
        
        
        if spec_enum == DsSpecs.SPEC_2_1_1:
            description = "Using european price in inputs"
            return DsSpec(input_cols=['S','K','r','q','sigma','T','EU_Price'],
                   output_cols=['Premium'],
                   gen_ranges={'S':(80,120),
                               'K':(80,120),
                               'r':(0.01,0.03),
                               'q':(0,0.03),
                               'sigma':(0.05,0.5),
                               'T':(1/12,3)},
                   get_input_func=InputFuncs.get_input_func_2,
                   get_output_func=OutputFuncs.get_output_func_1,
                   desc=description)

        if spec_enum == DsSpecs.SPEC_1_2_1:
            description = "Fixing K = 1"
            return DsSpec(input_cols=['S','r','q','sigma','T'],
                          output_cols=['Premium'],
                          gen_ranges={
                                  'S':(0.8,1.2),
                                  'r':(0.01,0.03),
                                  'q':(0,0.03),
                                  'sigma':(0.05,0.5),
                                  'T':(1/12,3)},
                         get_input_func=InputFuncs.get_input_func_1,
                         get_output_func=OutputFuncs.get_output_func_2,
                         desc=description)
            
        if spec_enum == DsSpecs.SPEC_3_3_1:
            description = "Fixing K = 1, Using EU_Price in inputs"
            return DsSpec(input_cols=['S','r','q','sigma','T','EU_Price'],
                          output_cols=['Premium'],
                          gen_ranges={
                                  'S':(0.8,1.2),
                                  'r':(0.01,0.03),
                                  'q':(0,0.03),
                                  'sigma':(0.05,0.5),
                                  'T':(1/12,3)},
                         get_input_func=InputFuncs.get_input_func_3,
                         get_output_func=OutputFuncs.get_output_func_3,
                         desc=description)                  