from dataset import DataSetKey
from model import NNModel
import os
from sklearn.preprocessing import StandardScaler
from my_config import Config
import pandas as pd

class Experiment:
    def __init__(self,data_set,model,train_size):
        self.data_set = data_set
        self.model = model
        self.train_size = train_size
        self.x_scaler = None
        self.y_scaler = None
        self.x_train = None
        self.y_train = None
    
    def _real_init(self):
        if self.x_scaler is None:
            raw_x,raw_y = self.data_set.get_data(self.train_size)
            self.x_scaler = StandardScaler()
            self.y_scaler = StandardScaler()
            self.x_train = self.x_scaler.fit_transform(raw_x)
            self.y_train = self.y_scaler.fit_transform(raw_y)
        
    @classmethod
    def load(cls,exp_dir,ds_manager):
        if not Config.FS.dir_exists(exp_dir):
            raise ValueError("Cannot load exp as {} not found".format(exp_dir))
        model = NNModel.load(os.path.join(exp_dir,'model'))
        dskey = DataSetKey.from_file(os.path.join(exp_dir,'dskey.p'))
        props = Config.FS.pickle_load(os.path.join(exp_dir,'props.p'))
        train_size = props['train_size']
        data_set = ds_manager.get_ds(dskey)
        return cls(data_set,model,train_size)
            
    @classmethod
    def create(cls,data_set,model_key,exp_dir,train_size):
        if Config.FS.dir_exists(exp_dir):
            raise ValueError("Cannot create new exp in existing dir {}".format(exp_dir))
        Config.FS.mkdirs(exp_dir)
        data_set.dskey.to_file(os.path.join(exp_dir,'dskey.p'))
        model = NNModel.create(data_set.spec,model_key,os.path.join(exp_dir,'model'))
        props = {'train_size':train_size}
        Config.FS.pickle_dump(props,os.path.join(exp_dir,'props.p'))
        return cls(data_set,model,train_size)
        
    def run(self,epochs):
        self._real_init()
        self.model.train((self.x_train,self.y_train),epochs)
    
    def get_metric(self,metric,metric_data,at_epoch=None):
        self._real_init()
        metric_x_raw,metric_y_raw = metric_data
        metric_x = self.x_scaler.transform(metric_x_raw)
        metric_y = self.y_scaler.transform(metric_y_raw)
        return self.model.get_metric(metric,(metric_x,metric_y),at_epoch)
    
    def get_pred(self,x_raw,at_epoch=None):
        self._real_init()
        x_pred = self.x_scaler.transform(x_raw)
        y_hat = self.model.predict(x_pred,at_epoch)
        return self.y_scaler.inverse_transform(y_hat)
                
    def epochs(self):
        return self.model.tot_epochs()
    
    def get_key(self):
        return (self.data_set.dskey.get_dict_key(),self.model.model_key,self.train_size)
    
    def get_info_df(self):
        ds_info = self.data_set.dskey.get_info_df()
        model_info = self.model.get_info_df()
        my_df = pd.concat([ds_info.reset_index(drop=True),
                           model_info.reset_index(drop=True)],axis=1)
        my_df['train_size'] = self.train_size
        return my_df
    
    def get_insample_data(self,num=None):
        num = self.train_size if num is None else num
        if num > self.train_size:
            raise ValueError("cannot get more data than train_size" +
                             "(requested:{},train_size:{})".format(num,self.train_size))
        return self.data_set.get_data(num)
    
    def get_outsample_data(self,num):
        if self.train_size + num > self.data_set.num_outputs():
            raise ValueError("num out sample requested plus train size " + 
                             "exceeds number of data in dataset" +
                             "(train_size:{},requested:{},available:{})".format(
                                     self.train_size,num,self.data_set.num_outputs()))
        return self.data_set.get_data(num,self.train_size)
        
        