import pandas as pd
from my_config import Config
from experiment import Experiment
import os
from multiprocessing import Process
import multiprocessing


class Runner:
    @staticmethod
    def run(exp,epochs):
        exp.run(epochs)

class ExperimentManager():
    def __init__(self,expm_dir,ds_man):
        self.expm_dir = expm_dir
        self.ds_man = ds_man
        
    @classmethod
    def load(cls,expm_dir,ds_man):
        if not Config.FS.dir_exists(expm_dir):
            raise ValueError("Cannot load exp as {} not found".format(expm_dir))
        return cls(expm_dir,ds_man)
    
    @classmethod
    def create(cls,expm_dir,ds_man):
        if Config.FS.dir_exists(expm_dir):
            raise ValueError("Cannot create new exp in existing dir {}".format(expm_dir))
        Config.FS.mkdirs(expm_dir)
        my_exps = {}
        Config.FS.pickle_dump(my_exps,os.path.join(expm_dir,'my_exps.p'))
        return cls(expm_dir,ds_man)
    
    @classmethod
    def create_or_load(cls,expm_dir,ds_man):
        if Config.FS.dir_exists(expm_dir):
            return cls.load(expm_dir,ds_man)
        else:
            return cls.create(expm_dir,ds_man)

    def new_exp(self,dskey,model_key,train_size):
        data_set = self.ds_man.get_ds(dskey)
        my_exps = Config.FS.pickle_load(os.path.join(self.expm_dir,'my_exps.p'))
        new_exp_key = (dskey.get_dict_key(),model_key,train_size)
        if new_exp_key in my_exps:
            raise ValueError("Attempting to add the same experiment {}".format(new_exp_key))
        exp_dir = os.path.join(self.expm_dir,"exp_{}".format(len(my_exps)))
        Experiment.create(data_set,model_key,exp_dir,train_size)
        my_exps[new_exp_key] = exp_dir
        Config.FS.pickle_dump(my_exps,os.path.join(self.expm_dir,'my_exps.p'))
    
    def get_info_df(self):
        all_exps = Config.FS.pickle_load(os.path.join(self.expm_dir,'my_exps.p'))
        if len(all_exps) == 0:
            return pd.DataFrame()
        def get_exp_df(exp_dir):
            exp = Experiment.load(exp_dir,self.ds_man)
            exp_df = exp.get_info_df()
            exp_df['exp_dir'] = exp_dir
            return exp_df
        all_exp_dfs = [get_exp_df(exp_dir) for key,exp_dir in all_exps.items()]
        return pd.concat(all_exp_dfs,axis=0,sort=False).reset_index(drop=True)
        
    def get_exp(self,dskey,model_key,train_size):
        all_exps = Config.FS.pickle_load(os.path.join(self.expm_dir,'my_exps.p'))
        exp_dir = all_exps[(dskey.get_dict_key(),model_key,train_size)]
        exp = Experiment.load(exp_dir,self.ds_man) 
        return exp
    
    def get_all_exp(self):
        all_exps = Config.FS.pickle_load(os.path.join(self.expm_dir,'my_exps.p'))
        all_exps = [Experiment.load(v,self.ds_man) for k,v in all_exps.items()]
        return all_exps
    
    @staticmethod
    def run_exps_multithread0(exps,epochs):
        
        processes = [Process(target=Runner.run,args=(exp,epoch))
                     for exp,epoch in zip(exps,epochs)]
        for proc in processes:
            proc.start()
        for proc in processes:
            proc.join()
    
    @staticmethod
    def run_exps_multithread(exps,epochs,num_processors=96):
        tasks = [(exp,epoch) for exp,epoch in zip(exps,epochs)]
        with multiprocessing.Pool(processes = num_processors) as p:
            p.starmap(ExperimentManager.exps_worker, [(i, tasks, num_processors) for i in range(num_processors)])
    
    @staticmethod
    def exps_worker(i, tasks, num_processors):
        num_tasks = len(tasks)
        work_load = int(num_tasks / num_processors)
        if i == num_processors-1:
            start, end = i*work_load, num_tasks
        else:
            start, end = i*work_load, (i+1)*work_load
        sub_tasks = tasks[start:end]
        for exp,epoch in sub_tasks:
            exp.run(epoch)