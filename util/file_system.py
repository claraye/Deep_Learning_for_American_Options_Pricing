from abc import ABC,abstractmethod
import pandas as pd
import os
import pickle
import tensorflow as tf
try:
    from datalab.context import Context
    import google.datalab.storage as storage
    from io import BytesIO
except ImportError:
    print('Not Running In Google Datalab')

class FileSystem(ABC):
    @abstractmethod
    def pickle_load(self,path):
        pass
    
    @abstractmethod
    def pickle_dump(self,obj,path):
        pass

    @abstractmethod
    def path_exists(self,path):
        pass
    
    @abstractmethod
    def dir_exists(self,path):
        pass
    
    @abstractmethod
    def mkdirs(self,path_to_dir):
        pass
    
    @abstractmethod
    def pd_read_csv(self,path,skiprows=None,nrows=None):
        pass
    
    @abstractmethod
    def pd_to_csv(self,path,df,index=True,header=True,mode='w'):
        pass
    
    @abstractmethod
    def new_placeholder_file(self,path):
        pass

    @abstractmethod
    def save_tf_sess(self,saver,sess,path):
        pass
    
    @abstractmethod
    def restore_tf_sess(self,saver,sess,path):
        pass
    
class WindowsFS(FileSystem):
    def __init__(self):
        pass
    
    def pickle_load(self,path):
        with open(path,'rb') as fs:
            return pickle.load(fs)
    
    def pickle_dump(self,obj,path):
        with open(path,'wb') as fs:
            pickle.dump(obj,fs)
    
    def path_exists(self,path):
        return os.path.exists(path)
   
    def dir_exists(self,path):
        return os.path.isdir(path)
    
    def mkdirs(self,path_to_dir):
        os.makedirs(path_to_dir)
    
    def pd_read_csv(self,path,skiprows=None,nrows=None):
        return pd.read_csv(path,skiprows=skiprows,nrows=nrows)
    
    def pd_to_csv(self,path,df,index=True,header=True,mode='w'):
        df.to_csv(path,index=index,header=header,mode=mode)
        
    def new_placeholder_file(self,path):
        open(path,'a')
    
    def save_tf_sess(self,sess,model_dir,epoch):
        saver = tf.compat.v1.train.Saver()
        model_path = os.path.join(model_dir,'model')
        saver.save(sess,model_path,global_step=epoch)
        
    def restore_tf_sess(self,sess,model_dir,epoch=None):
        if epoch is None:
            cp_path = tf.train.latest_checkpoint(model_dir)
        else:
            cp_path = os.path.join(model_dir,'model-{}'.format(epoch))
        saver = tf.train.import_meta_graph(cp_path + '.meta')
        saver.restore(sess,cp_path)
    
    
class GcsFS(FileSystem):    
    def __init__(self):
        self.bucket_name = None
    
    def set_bucket_name(self,bucket_name):
        self.bucket_name = bucket_name
    
    def pickle_load(self,path):
        mybucket = storage.Bucket(self.bucket_name)
        remote_pickle = mybucket.object(path).read_stream()
        return pickle.load(BytesIO(remote_pickle))
    
    def pickle_dump(self,obj,path):
        local_pkl_name = os.path.split(path)[-1]
        # Create a local pickle file
        with open(local_pkl_name,'wb') as fs:
            pickle.dump(obj,fs)
        
        # Define storage bucket
        mybucket = storage.Bucket(self.bucket_name)
        # Create storage bucket if it does not exist
        if not mybucket.exists():
            mybucket.create()        

        # Write pickle to GCS
        tem_object = mybucket.object(path)
        with open(local_pkl_name, 'rb') as f:
            tem_object.write_stream(bytearray(f.read()), 'application/octet-stream')
        
    def path_exists(self,path):
        filtered_paths = [o.key for o in storage.Bucket(self.bucket_name).objects() 
                          if o.key.startswith(path)]
        return len(filtered_paths) != 0
   
    def dir_exists(self,path):
        # only return True if that path exists and is a directory
        adj_path = os.path.join(path,'')   # make sure that the adj_path is ended with '/'
        filtered_paths = [o.key for o in storage.Bucket(self.bucket_name).objects() 
                          if o.key.startswith(adj_path)]
        return len(filtered_paths) != 0
    
    def mkdirs(self,path_to_dir):
        placeholder_path = os.path.join(path_to_dir, 'placeholer.txt')
        self.new_placeholder_file(placeholder_path)
    
    def pd_read_csv(self,path,skiprows=None,nrows=None):
        mybucket = storage.Bucket(self.bucket_name)
        remote_file = mybucket.object(path).read_stream()
        return pd.read_csv(BytesIO(remote_file),skiprows=skiprows,nrows=nrows)
    
    def pd_to_csv(self,path,df,index=True,header=True,mode='w'): 
        local_csv_name = os.path.split(path)[-1]
        if mode == 'a' and self.path_exists(path):
            try:
                original_df = self.pd_read_csv(path,header=None)    # treat the header (if any) as part of dataset
                df.columns = original_df.columns    # to ignore the header of df
                df = pd.concat([original_df, df], ignore_index=True)    # append to the end of original_df

                # Create a local pickle file, with aggregated data
                df.to_csv(local_csv_name,index=index,header=False,mode='w')
            except pd.errors.EmptyDataError:
                # if the file exsits but is empty, append file is equivalent to write
                df.to_csv(local_csv_name,index=index,header=header,mode='w')
        else:
            # Create a local pickle file
            df.to_csv(local_csv_name,index=index,header=header,mode=mode)
        
        # Define storage bucket
        mybucket = storage.Bucket(self.bucket_name)
        # Create storage bucket if it does not exist
        if not mybucket.exists():
            mybucket.create()
        
        # Write csv to GCS using pickle
        tem_object = mybucket.object(path)
        with open(local_csv_name, 'rb') as f:
            tem_object.write_stream(bytearray(f.read()), 'application/octet-stream')
        
    def new_placeholder_file(self,path):
        state = ''
        self.pickle_dump(state,path)

    def save_tf_sess(self,sess,model_dir,epoch):
        saver = tf.compat.v1.train.Saver()
        model_dir = os.path.join('gs://'+ self.bucket_name, model_dir)
        model_path = os.path.join(model_dir,'model')
        saver.save(sess,model_path,global_step=epoch)
        
    def restore_tf_sess(self,sess,model_dir,epoch=None):
        model_dir = os.path.join('gs://'+ self.bucket_name, model_dir)
        if epoch is None:
            cp_path = tf.train.latest_checkpoint(model_dir)
        else:
            cp_path = os.path.join(model_dir,'model-{}'.format(epoch))
        saver = tf.train.import_meta_graph(cp_path + '.meta')
        saver.restore(sess,cp_path)