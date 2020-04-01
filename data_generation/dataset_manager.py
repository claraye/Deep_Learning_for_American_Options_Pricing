from abc import ABC,abstractmethod
from dataset import DataSetKey,FileBasedDataSet
from ds_spec import DsSpecs
from my_enums import Pricers,Generators
import os
import sys
sys.path.append('..')
from my_config import Config
import pandas as pd 


class DataSetManager(ABC):
    @abstractmethod
    def get_all_ds(self):
        pass

    @abstractmethod
    def get_ds(self,key):
        pass
    
    @staticmethod
    def get_default_in_mem_DSM():
        dskeys = [DataSetKey(gen_enum,am_pricer_enum,eu_pricer_enum,spec_enum)
                    for spec_enum in DsSpecs 
                    for gen_enum in Generators
                    for am_pricer_enum in Pricers.american_pricers()
                    for eu_pricer_enum in Pricers.european_pricers()]
        
        datasets = {
                key.get_dict_key() : 
                    FileBasedDataSet.create_or_load(
                            key,os.path.join(Config.datasets_dir,"ds_{}".format(i))) 
                for i,key in enumerate(dskeys)
                }
        return InMemoryDataSetManager(datasets)
    
    @staticmethod
    def get_set1_in_mem_DSM():
        # get datasets generated with: 
        # Halton qmc sequence; Juzhong/Whaley approximation; Fixing K = 1, Using EU_Price in inputs
        dskey = DataSetKey(Generators.Halton,Pricers.JuZhongWhaley,Pricers.BSFormula,DsSpecs.SPEC_3_3_1)
        datasets = {
            dskey.get_dict_key():FileBasedDataSet.load(os.path.join(Config.datasets_dir,"ds_23"))
        }
        return InMemoryDataSetManager(datasets)
    
    @staticmethod
    def get_set2_in_mem_DSM():
        # get datasets generated with: 
        # Uniform random generator; Juzhong/Whaley approximation; Fixing K = 1, Using EU_Price in inputs
        dskey = DataSetKey(Generators.Uniform,Pricers.JuZhongWhaley,Pricers.BSFormula,DsSpecs.SPEC_3_3_1)
        datasets = {
            dskey.get_dict_key():FileBasedDataSet.load(os.path.join(Config.datasets_dir,"ds_20"))
        }
        return InMemoryDataSetManager(datasets)
    
    @staticmethod
    def get_set3_in_mem_DSM():
        # get datasets generated with: 
        # Halton qmc sequence; Juzhong/Whaley approximation; Fixing K = 1, But not including EU_Price in inputs
        dskey = DataSetKey(Generators.Halton,Pricers.JuZhongWhaley,Pricers.BSFormula,DsSpecs.SPEC_1_2_1)
        datasets = {
            dskey.get_dict_key():FileBasedDataSet.load(os.path.join(Config.datasets_dir,"ds_17"))
        }
        return InMemoryDataSetManager(datasets)
    
    @abstractmethod
    def get_info_df(self):
        pass
    
#For simplicity we save all datasets inMemory
#If needed, make FileDataSetManager(DataSetManager) to save Datasets in a file
class InMemoryDataSetManager(DataSetManager):
    def __init__(self,datasets):
        self.datasets = datasets
        
    def get_ds(self,key):
        dict_key = key.get_dict_key()
        if dict_key in self.datasets:
            return self.datasets[dict_key];
        else:
            raise ValueError("No Dataset has key: {}".format(key))
    
    def get_all_ds(self):
        return self.datasets
    
    def get_info_df(self):
        return pd.concat([v.get_info_df() for k,v in self.datasets.items()],
                          axis=0,sort=False).reset_index(drop=True)
