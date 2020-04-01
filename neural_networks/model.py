import numpy as np
import pandas as pd
import tensorflow as tf
from nn_builder import build_NN,restore_NN
import sys
sys.path.append('../util')
from my_config import Config
import os


class NNModel:
    def __init__(self,model_key,model_dir):
        self.model_key = model_key
        #self.graph = graph
        #self.nn = nn
        #self.sess = sess
        self.model_dir = model_dir
        
        
    @classmethod
    def create(cls,ds_spec,model_key,model_dir):
        #all_files = FileBasedDataSet.all_files(ds_dir) 
        #inputs_file,outputs_file,key_file,generator_file = all_files
        if Config.FS.dir_exists(model_dir):
            raise ValueError("Cannot create new NN in existing dir {}".format(model_dir))
        Config.FS.mkdirs(model_dir)
        graph = tf.Graph()
        layers,learn_rate = model_key
        with graph.as_default():
            build_NN(layers,ds_spec.num_input_cols(),learn_rate)
            sess = tf.compat.v1.Session()
            sess.run(tf.compat.v1.global_variables_initializer())
            Config.FS.save_tf_sess(sess,model_dir,0)
        Config.FS.pickle_dump(model_key,os.path.join(model_dir,'model_key.p'))
        return cls(model_key,model_dir)#,graph,nn,sess)
    
    @classmethod
    def load(cls,model_dir):
        if not Config.FS.dir_exists(model_dir):
            raise ValueError("Cannot load NN as {} does not exists".format(model_dir))
        #graph = tf.Graph()
        #with graph.as_default():
        #    sess = tf.compat.v1.Session()
        #    Config.FS.restore_tf_sess(sess,model_dir,None)
        #    nn = restore_NN()
        model_key = Config.FS.pickle_load(os.path.join(model_dir,'model_key.p'))
        return cls(model_key,model_dir)#,graph,nn,sess)    
        
    @classmethod
    def create_or_load(cls,ds_spec,model_key,model_dir):
        if Config.FS.dir_exists(model_dir):
            print("Loading {} at {}".format(model_key,model_dir))
            ret = cls.load(model_dir)
            if ret.model_key != model_key:
                raise ValueError("given model_key = {}, model_key from dir = {}".format(
                        model_key,ret.model_key))
            return ret
        else:
            print("Creating {} at {}".format(model_key,model_dir))
            return cls.create(ds_spec,model_key,model_dir)
        
    def train(self,train_data,epochs,batch_size=1000):
        t_graph = tf.Graph()
        train_x,train_y = train_data
        with t_graph.as_default():
            with tf.compat.v1.Session() as sess:
                Config.FS.restore_tf_sess(sess,self.model_dir,None)
                x,y,is_train,output,cost,train_op,epoch_num = restore_NN()
                inc = tf.assign_add(epoch_num, 1)
                for epoch in range(epochs):
                    total_batch = int(train_x.shape[0]/batch_size)
                    for i in range(total_batch):
                        start,end = i*batch_size,(i+1)*batch_size
                        batch_x,batch_y = train_x[start:end],train_y[start:end]
                        sess.run(train_op,feed_dict={x:batch_x,y:batch_y,is_train:True})
                    sess.run(inc)
                    Config.FS.save_tf_sess(sess,self.model_dir,sess.run(epoch_num))
            
    def predict(self,pred_x,at_epoch=None):
        p_graph = tf.Graph()
        #pred_x,pred_y = pred_data
        with p_graph.as_default():
            with tf.compat.v1.Session() as sess:
                Config.FS.restore_tf_sess(sess,self.model_dir,at_epoch)
                nn = restore_NN()
                x,y,is_train,output,cost,train_op,epoch_num = nn
                pred_y = np.zeros((pred_x.shape[0],y.shape[1].value))
                return sess.run(output,feed_dict={x:pred_x,y:pred_y,is_train:False})        
        
    def get_metric(self,metric_f,metric_data,at_epoch=None,is_train_val=False):
        p_graph = tf.Graph()
        metric_x,metric_y = metric_data
        with p_graph.as_default():
            with tf.compat.v1.Session() as sess:
                Config.FS.restore_tf_sess(sess,self.model_dir,at_epoch)
                nn = restore_NN()
                x,y,is_train,output,cost,train_op,epoch_num = nn
                metric = metric_f(nn)
                return sess.run(metric,feed_dict={x:metric_x,y:metric_y,is_train:is_train_val})
    
    def tot_epochs(self):
        graph = tf.Graph()
        with graph.as_default():
            with tf.compat.v1.Session() as sess:
                Config.FS.restore_tf_sess(sess,self.model_dir,None)
                x,y,is_train,output,cost,train_op,epoch_num = restore_NN()
                return sess.run(epoch_num)
    
    def get_key(self):
        return self.model_key
    
    def get_info_df(self):
        layers,lr = self.model_key
        nodes = [n for n,af,dr in layers]
        afs = [af.__name__ for n,af,dr in layers]
        drs = [dr for n,af,dr in layers]
        epochs = self.tot_epochs()
        ret = pd.DataFrame({'nodes':[nodes],'Activation Funcs':[afs],
                            'drop rates':[drs],'epochs run':[epochs]})
        return ret