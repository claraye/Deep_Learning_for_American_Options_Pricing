import tensorflow as tf

def layer(x,weight_shape,bias_shape,drop_rate,activation_func=tf.identity):
    tf.identity(x,name='input')
    w_std = (2.0/weight_shape[0]) ** 0.5
    w_0 = tf.random_normal_initializer(stddev=w_std)
    b_0 = tf.constant_initializer(value=0)
    W = tf.compat.v1.get_variable("W",weight_shape,initializer=w_0)
    b = tf.compat.v1.get_variable("b",bias_shape,initializer=b_0)
    ret = tf.identity(activation_func((tf.matmul(x,W) + b)/tf.sqrt(tf.reduce_sum(W*W,axis=0))),name='ret')
    ret = tf.nn.dropout(ret,rate=drop_rate)
    return ret

def build_layers(x,layer_nodes,layer_names,activation_funcs,drop_rates):
    layer_io = zip(layer_nodes,layer_nodes[1:])
    curr = x
    for (in_nodes,out_nodes),layer_name,f,dr in zip(layer_io,layer_names,activation_funcs,drop_rates):
        with tf.compat.v1.variable_scope(layer_name):
            curr = layer(curr,[in_nodes,out_nodes],[out_nodes],dr,f)
    return curr

def inference(x,hidden_layer_nodes,num_output,hidden_activation_funcs,hidden_layer_drop_rates,output_activation_func):
    layer_nodes = [int(x.shape[1])] + hidden_layer_nodes + [num_output]
    layer_names = ['hidden_layer_{}'.format(i+1) for i in range(len(hidden_layer_nodes))] + ['output']
    activation_funcs = hidden_activation_funcs + [output_activation_func]
    drop_rates = hidden_layer_drop_rates + [0]
    output = build_layers(x,layer_nodes,layer_names,activation_funcs,drop_rates)
    return output

def evaluate(output,y):
    correct_prediction = (output*y) > 0
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    tf.compat.v1.summary.scalar("validation_error",(1.0 - accuracy))
    return accuracy

def training(cost, global_step,learning_rate):
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cost,global_step=global_step,name='train')
    return train_op

def loss(output,y):
    return tf.reduce_mean(tf.square(tf.subtract(output,y)),name='loss')

def build_NN(hidden_layers,input_dim,learning_rate):
    output_dim = 1
    
    x = tf.compat.v1.placeholder("float",[None,input_dim],name='x')
    y = tf.compat.v1.placeholder("float",[None,output_dim],name='y')
    is_train = tf.compat.v1.placeholder(tf.bool,name='is_train')
    hidden_layer_nodes = [n for n,a_f,d_r in hidden_layers]
    hidden_activation_funcs = [a_f for n,a_f,d_r in hidden_layers]
    output_activation_func = tf.identity
    hidden_drop_rates = [float(d_r) for n,a_f,d_r in hidden_layers]
    hidden_drop_rates = [tf.cond(is_train,lambda:d_r,lambda:0.0) 
                         for d_r in hidden_drop_rates]
    output = inference(x,hidden_layer_nodes,output_dim,
                       hidden_activation_funcs,hidden_drop_rates,
                       output_activation_func)
    cost = loss(output,y)
    global_step = tf.Variable(0,name='global_step',trainable=False)
    epoch_num = tf.Variable(0,name='epoch_num',trainable=False)
    train_op = training(cost,global_step,learning_rate)
    return x,y,is_train,output,cost,train_op,epoch_num

def restore_NN():
    graph = tf.compat.v1.get_default_graph()
    x = graph.get_tensor_by_name('x:0')
    y = graph.get_tensor_by_name('y:0')
    output = graph.get_tensor_by_name('output/ret:0')
    cost = graph.get_tensor_by_name('loss:0')
    train_op = graph.get_tensor_by_name('train:0')
    epoch_num = graph.get_tensor_by_name('epoch_num:0')
    is_train = graph.get_tensor_by_name('is_train:0')
    return x,y,is_train,output,cost,train_op,epoch_num