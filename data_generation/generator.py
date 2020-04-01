from abc import ABC,abstractmethod
import numpy as np
import pickle
'''
import ghalton
halton_method = 'ghalton'
'''
try:
    import ghalton
    halton_method = 'ghalton'
except ImportError:
    from halton_qmc import halton_sequence
    halton_method = 'halton_qmc'

import random
import os
import sys
sys.path.append('..')
from my_config import Config

class Generator(ABC):
    @abstractmethod
    def generate(self,cols,ranges,num):
        pass
    
    @abstractmethod
    def load(self,file):
        pass
    
    @abstractmethod
    def save_state(self,file):
        pass
    
class UniformGenerator(Generator):
    def __init__(self):
        pass
    
    def generate(self,cols,ranges,num):
        inputs =[np.random.uniform(low=ranges[col][0],high=ranges[col][1],size=num)
                 for col in cols]
        return np.vstack(inputs).T
    
    def load(self,file):
        return
    
    def save_state(self,file):
        Config.FS.new_placeholder_file(file)


class HaltonGenerator(Generator):
    def __init__(self):
        self.tot_num = 0
        self.seed = random.randint(0,100)
        
    def generate(self,cols,ranges,num): 
        global halton_method
        if halton_method == 'ghalton':
            # use the ghalton package
            sequencer = ghalton.Halton(len(cols))
            sequencer.seed(self.seed)
            sequencer.get(self.tot_num)
            seq = sequencer.get(num)
            self.tot_num += num
            low = np.asarray([ranges[col][0] for col in cols])
            high = np.asarray([ranges[col][1] for col in cols])
            return low + (high - low) * seq
        elif halton_method == 'halton_qmc':
            # use the halton_qmc module
            seq = halton_sequence(self.tot_num+1, self.tot_num+num, len(cols)).reshape(num,-1)
            self.tot_num += num
            low = np.asarray([ranges[col][0] for col in cols])
            high = np.asarray([ranges[col][1] for col in cols])
            return low + (high - low) * seq
        
    def load(self,file):
        state = Config.FS.pickle_load(file)
        self.tot_num = state['tot_num']
        self.seed = state['seed']
            
    def save_state(self,file):
        state = {'tot_num':self.tot_num,'seed':self.seed}
        Config.FS.pickle_dump(state,file)