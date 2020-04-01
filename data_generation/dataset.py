from abc import ABC,abstractmethod
import numpy as np
import pandas as pd
import multiprocessing
import os
import sys
sys.path.append('..')

from my_config import Config

from ds_spec import DsSpecs
from my_enums import Generators,Pricers


class DataSetKey:
    def __init__(self,generator_enum,am_pricer_enum,
                 eu_pricer_enum,spec_enum):
        self.ge = generator_enum
        self.am_pe = am_pricer_enum
        self.eu_pe = eu_pricer_enum
        self.se = spec_enum
    
    def get_dict_key(self):
        return (self.ge,self.am_pe,self.eu_pe,self.se)
    
    def __repr__(self):
        return "DataSetKey({}, {}, {}, {})".format(self.ge,self.am_pe,
                          self.eu_pe,self.se)
        
    @classmethod
    def from_file(cls,f):
        return Config.FS.pickle_load(f)
        
    def to_file(self,f):
        Config.FS.pickle_dump(self,f)
    
    def get_info_df(self):
        return pd.DataFrame({
                'Generator':[self.ge],
                'American Pricer':[self.am_pe],
                'European Pricer':[self.eu_pe],
                'Specs':[self.se]})
    
class DataSet(ABC):
    def __init__(self,dskey):
        self.dskey = dskey;
        self.generator = Generators.from_enum(dskey.ge)
        self.am_pricer = Pricers.from_enum(dskey.am_pe)
        self.eu_pricer = Pricers.from_enum(dskey.eu_pe)
        self.spec = DsSpecs.from_enum(dskey.se)
        self.data_makers = (self.generator,self.am_pricer,self.eu_pricer)
        
    @abstractmethod
    def num_inputs(self):
        pass
    
    @abstractmethod
    def num_outputs(self):
        pass
    
    def num(self):
        return self.num_outputs()
   
    @abstractmethod     
    def _save_new_inputs(self,new_inputs):
        pass
    
    def incr_inputs(self,num):
        new_inputs = self.spec.get_input_func(self.data_makers)(num)
        self._save_new_inputs(new_inputs)
    
    def incr_inputs_multithread(self,num,num_threads):        
        with multiprocessing.Pool(processes = num_threads) as p:
            sub_new_inputs = p.starmap(self.incr_inputs_worker, [(i, num, num_threads) for i in range(num_threads)])      
        new_inputs = np.concatenate([sub_new_inputs[i] for i in range(len(sub_new_inputs))])
        self._save_new_inputs(new_inputs)
        #return new_inputs

    def incr_inputs_worker(self, i, num, num_threads):
        # Helper function for incr_inputs_multithread(); to run a single thread
        batch_size = int(num/num_threads)
        if i == num_threads-1:
            sub_num = num - i*batch_size
        else:
            sub_num = batch_size
        sub_new_inputs = self.spec.get_input_func(self.data_makers)(sub_num)
        return sub_new_inputs
    
    
    @abstractmethod
    def _save_new_outputs(self,new_outputs):
        pass
    
    def incr_outputs(self,num):
        if self.num_outputs() + num > self.num_inputs():
            self.incr_inputs(self.num_outputs() + num - self.num_inputs())
            self.incr_outputs(num)
        else:
            inputs_used = self._get_inputs(num,self.num_outputs())
            new_outputs = self.spec.get_output_func(self.data_makers)(inputs_used)
            self._save_new_outputs(new_outputs)
           
    def incr_outputs_multithread(self,num,num_threads):
        if self.num_outputs() + num > self.num_inputs():
            self.incr_inputs(self.num_outputs() + num - self.num_inputs())
            self.incr_outputs_multithread(num, num_threads)
        else:
            inputs_used = self._get_inputs(num,self.num_outputs())
            
            with multiprocessing.Pool(processes = num_threads) as p:
                sub_new_outputs = p.starmap(self.incr_outputs_worker, [(i, num, num_threads, inputs_used) for i in range(num_threads)])      
            new_outputs = np.concatenate([sub_new_outputs[i] for i in range(len(sub_new_outputs))])
            self._save_new_outputs(new_outputs)
            #return new_outputs

    def incr_outputs_worker(self, i, num, num_threads, inputs_used):
        # Helper function for incr_outputs_multithread(); to run a single thread
        batch_size = int(num/num_threads)
        if i == num_threads-1:
            start, end = i*batch_size, num
        else:
            start, end = i*batch_size, (i+1)*batch_size
        sub_inputs_used = inputs_used[start:end]
        sub_new_outputs = self.spec.get_output_func(self.data_makers)(sub_inputs_used)
        return sub_new_outputs

    @abstractmethod
    def _get_inputs(self,num,start):
        pass
    
    @abstractmethod
    def _get_outputs(self,num,start):
        pass
    
    def get_data(self,num,start=0):
        if start + num > self.num_outputs():
            raise ValueError("Not Enough Data" +
            "(Requested: {}, Inputs #: {}, Outputs #: {})".format(
            start + num,self.num_inputs(),self.num_outputs()))
        inputs = self._get_inputs(num,start)
        outputs = self._get_outputs(num,start)
        return (inputs,outputs)
    
    def get_data_df(self,num,start=0):
        inputs,outputs = self.get_data(num,start)
        data = np.append(inputs,outputs,axis=1)
        cols = self.spec.input_cols() + self.spec.output_cols()
        return pd.DataFrame(data,columns=cols)
    
    def validate(self):
        if self.num_outputs() > self.num_inputs():
            raise RuntimeError("Invalid Dataset: # of outputs greater than # of inputs")
        if self._get_inputs(0,0).shape[1] != self.spec.num_input_cols():
            raise RuntimeError("Invalid Dataset: incorrect input shape")
        if self._get_outputs(0,0).shape[1] != self.spec.num_output_cols():
            raise RuntimeError("Invalid Dataset: incorrect output shape")
    
    def get_info_df(self):
        df = self.dskey.get_info_df()
        df['Spec Desc'] = self.spec.desc
        df['Input Cols'] = str(self.spec.input_cols())
        df['Output Cols'] = str(self.spec.output_cols())
        df['Num Inputs'] = self.num_inputs()
        df['Num Outputs'] = self.num_outputs()
        return df
    
class InMemoryDataSet(DataSet):
    def __init__(self,dskey):
        super(InMemoryDataSet,self).__init__(dskey)
        self.inputs = np.zeros((0,self.spec.num_input_cols()));
        self.outputs = np.zeros((0,self.spec.num_output_cols()));
        
    def num_inputs(self):
        return self.inputs.shape[0]
    
    def num_outputs(self):
        return self.outputs.shape[0]

    def _save_new_inputs(self,new_inputs):
        self.inputs = np.append(self.inputs,new_inputs,axis=0)
    
    def _save_new_outputs(self,new_outputs):
        self.outputs = np.append(self.outputs,new_outputs,axis=0)

    def _get_inputs(self,num,start):
        return self.inputs[start:start+num]
        
    def _get_outputs(self,num,start):
        return self.outputs[start:start+num]

    
class FileBasedDataSet(DataSet):
    def __init__(self,dskey,ds_dir,num_inputs,num_outputs):
        super(FileBasedDataSet,self).__init__(dskey)
        self.ds_dir = ds_dir
        inputs_file,outputs_file,key_file,generator_file = FileBasedDataSet.all_files(ds_dir)
        self.inputs_file = inputs_file
        self.outputs_file = outputs_file
        self.key_file = key_file
        self.generator_file = generator_file
        self.generator.save_state(self.generator_file)
        self._num_inputs = num_inputs
        self._num_outputs = num_outputs
        self.generator.load(self.generator_file)
    
    @staticmethod
    def all_files(ds_dir):
        inputs_file = os.path.join(ds_dir,"inputs.csv")
        outputs_file = os.path.join(ds_dir,"outputs.csv")
        key_file = os.path.join(ds_dir,"key.p")
        generator_file = os.path.join(ds_dir,"gen.txt")
        return inputs_file,outputs_file,key_file,generator_file
    
    @classmethod
    def create(cls,dskey,ds_dir,input_data=None,output_data=None):
        all_files = FileBasedDataSet.all_files(ds_dir) 
        inputs_file,outputs_file,key_file,generator_file = all_files
        for f in all_files:
            if Config.FS.path_exists(f):
                raise ValueError("Cannot create new dataset as {} already exists".format(f))
        if not Config.FS.dir_exists(ds_dir):
            Config.FS.mkdirs(ds_dir)
        dskey.to_file(key_file)
        spec = DsSpecs.from_enum(dskey.se)
        Config.FS.pd_to_csv(
                inputs_file,
                pd.DataFrame(input_data,columns=spec.input_cols()),
                index=False)
        Config.FS.pd_to_csv(
                outputs_file,
                pd.DataFrame(output_data,columns=spec.output_cols()),
                index=False)
        return cls(dskey,ds_dir,0,0)
    
    @classmethod
    def load(cls,ds_dir):
        all_files = FileBasedDataSet.all_files(ds_dir)
        inputs_file,outputs_file,key_file,generator_file = all_files
        for f in all_files:
            if not Config.FS.path_exists(f):
                raise ValueError("Cannot load dataset as {} does not exist but folder exists".format(f))
        dskey = DataSetKey.from_file(key_file)
        num_inputs = Config.FS.pd_read_csv(inputs_file).shape[0]
        num_outputs = Config.FS.pd_read_csv(outputs_file).shape[0]
        ret = cls(dskey,ds_dir,num_inputs,num_outputs)
        try:
            ret.validate()
        except ValueError as e:
            raise ValueError("{}({})".format(e,ds_dir))
        return ret
    
    @classmethod
    def create_or_load(cls,dskey,ds_dir):
        if Config.FS.dir_exists(ds_dir):
            print("Loading {} at {}".format(dskey,ds_dir))
            ret = FileBasedDataSet.load(ds_dir)
            if ret.dskey.get_dict_key() != dskey.get_dict_key():
                raise "given dskey = {}, dskey from dir = {}".format(dskey,ret.dskey)
            return ret
        else:
            print("Creating {} at {}".format(dskey,ds_dir))
            return FileBasedDataSet.create(dskey,ds_dir)
        
    def num_inputs(self):
        return self._num_inputs
    
    def num_outputs(self):
        return self._num_outputs

    def _save_new_inputs(self,new_inputs):
        Config.FS.pd_to_csv(self.inputs_file,
                            pd.DataFrame(new_inputs),
                            mode='a',index=False,header=False)
        self.generator.save_state(self.generator_file)
        self._num_inputs += new_inputs.shape[0]
                
    def _save_new_outputs(self,new_outputs):
        Config.FS.pd_to_csv(self.outputs_file,
                            pd.DataFrame(new_outputs),
                            mode='a',index=False,header=False)
        self._num_outputs += new_outputs.shape[0]
        
    def _get_inputs(self,num,start):
        return Config.FS.pd_read_csv(self.inputs_file,skiprows=range(1,start+1),nrows=num).values
        
    def _get_outputs(self,num,start):
        return Config.FS.pd_read_csv(self.outputs_file,skiprows=range(1,start+1),nrows=num).values
    
    def get_info_df(self):
        df = super(FileBasedDataSet,self).get_info_df()
        df['location'] = self.ds_dir
        return df