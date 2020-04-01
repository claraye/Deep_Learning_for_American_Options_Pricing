import tensorflow as tf

class Metrics:
    @staticmethod
    def mse(nn):
        x,y,is_train,output,cost,train_op,epoch_num = nn
        return tf.reduce_mean(tf.square(tf.subtract(output,y))) 
    
    @staticmethod
    def r2(nn):
        x,y,is_train,output,cost,train_op,epoch_num = nn
        mse = Metrics.mse(nn)
        tss = tf.reduce_mean(tf.square(tf.subtract(y,tf.reduce_mean(y))))
        return tf.multiply(100.0,tf.subtract(1.0,tf.divide(mse,tss)))