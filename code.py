# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 10:50:29 2018

@author: LF
"""


from scipy.io import loadmat
import scipy.io
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error #mse
from sklearn.metrics import mean_absolute_error #mae
from sklearn.metrics import r2_score#R square

import os
dir_path = os.path.dirname(os.path.abspath(__file__))
print(dir_path)
TIME_STEPS = 125    # backpropagation through time 的 time_steps
#——————————————————导入数据——————————————————————
dataPPGtra=[]
dataABPtra=[]
#dataECGtra=[]
dataPPGtes=[]
dataABPtes=[]
#dataECGtes=[]
for file in os.listdir("./data/train/"):
    result_dict_1=loadmat(os.path.join(dir_path,"./data/train",file))
    for i in range(result_dict_1["p"][0].shape[0]):
        for j in range(int(result_dict_1["p"][0][i].shape[1]/TIME_STEPS)-1):
            scaler1=MinMaxScaler()
            scaler2=MinMaxScaler()
#            scaler3=MinMaxScaler()
            dataPPGtra.append(result_dict_1["p"][0][i][0,j*TIME_STEPS:(j+1)*TIME_STEPS])
            dataABPtra.append(result_dict_1["p"][0][i][1,j*TIME_STEPS:(j+1)*TIME_STEPS])#
#            dataECGtra.append(result_dict_1["p"][0][i][2,j*TIME_STEPS:(j+1)*TIME_STEPS])
for file in os.listdir("./data/test/"):
    result_dict_2 = loadmat(os.path.join(dir_path, "./data/test/", file))
    for i in range(result_dict_2["p"][0].shape[0]):
        for j in range(int(result_dict_2["p"][0][i].shape[1] / TIME_STEPS )- 1):
            dataPPGtes.append(result_dict_2["p"][0][i][0, j * TIME_STEPS:(j + 1) * TIME_STEPS])
            dataABPtes.append(result_dict_2["p"][0][i][1, j * TIME_STEPS:(j + 1) * TIME_STEPS])
#            dataECGtes.append(result_dict_2["p"][0][i][2, j * TIME_STEPS:(j + 1) * TIME_STEPS])
dataPPGtra=np.array(dataPPGtra)
dataABPtra=np.array(dataABPtra)
#dataECGtra=np.array(dataECGtra)
dataPPGtes=np.array(dataPPGtes)
dataABPtes=np.array(dataABPtes)
#dataECGtes=np.array(dataECGtes)

scaler1=MinMaxScaler()
scaler2=MinMaxScaler()
scaler3=MinMaxScaler()
scaler4=MinMaxScaler()
#scaler5=MinMaxScaler()
#scaler6=MinMaxScaler()

scaler1.fit(dataPPGtra)
dataPPGtra=scaler1.transform(dataPPGtra)

scaler2.fit(dataABPtra)
dataABPtra=scaler2.transform(dataABPtra)

#scaler3.fit(dataECGtra)
#dataECGtra=scaler3.transform(dataECGtra)

scaler3.fit(dataPPGtes)
dataPPGtes=scaler3.transform(dataPPGtes)

scaler4.fit(dataABPtes)
dataABPtes=scaler4.transform(dataABPtes)

#scaler6.fit(dataECGtes)
#dataECGtes=scaler6.transform(dataECGtes)
#dataABPtes=scaler4.inverse_transform(dataABPtes)
# dataPPGtra=result_dict_1['p'][0][:][0]


BATCH_START = 0     # 建立 batch data 时候的 index
TIME_STEPS = 125    # backpropagation through time 的 time_steps
BATCH_SIZE = 1
INPUT_SIZE = 1      # sin 数据输入 size
OUTPUT_SIZE = 1     # cos 数据输出 size
CELL_SIZE = 10      # RNN 的 hidden unit size 
LR = 0.006          # learning rate
print(dataABPtra.shape)
def get_batch(step):
    global BATCH_START, TIME_STEPS
    # xs shape (1642batch, 500steps)
    xs = np.arange(BATCH_START, BATCH_START+TIME_STEPS*BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS))
   
    seq = dataPPGtra[step:step+BATCH_SIZE,:]
    res = dataABPtra[step:step+BATCH_SIZE,:]

    BATCH_START += TIME_STEPS
    
    return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]

# =============================================================================
 
class LSTMRNN(object):
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size):
        #ValueError:https://sthsf.github.io/2017/06/18/ValueError:%20kernel%20already%20exists/
        tf.reset_default_graph()
        
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')
            self.ys = tf.placeholder(tf.float32, [None, n_steps, output_size], name='ys')
#error：http://blog.csdn.net/u014283248/article/details/64440268
        with tf.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.variable_scope('LSTM_cell'):
            self.add_cell()
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)
    def add_input_layer(self,):
        l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')  # (batch*n_step, in_size)
        
        Ws_in = self._weight_variable([self.input_size, self.cell_size])
        
        bs_in = self._bias_variable([self.cell_size,])
        
        with tf.name_scope('Wx_plus_b'):
            l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
        
        self.l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.cell_size], name='2_3D')
    def add_cell(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
            lstm_cell, self.l_in_y, initial_state=self.cell_init_state, time_major=False)
    def add_output_layer(self):
        
        l_out_x = tf.reshape(self.cell_outputs, [-1, self.cell_size], name='2_2D')
        Ws_out = self._weight_variable([self.cell_size, self.output_size])
        bs_out = self._bias_variable([self.output_size, ])
       
        with tf.name_scope('Wx_plus_b'):
            self.pred = tf.matmul(l_out_x, Ws_out) + bs_out
    def compute_cost(self):
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [tf.reshape(self.pred, [-1], name='reshape_pred')],
            [tf.reshape(self.ys, [-1], name='reshape_target')],
            [tf.ones([self.batch_size * self.n_steps], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.ms_error,
            name='losses'
        )
        with tf.name_scope('average_cost'):
            self.cost = tf.div(
                tf.reduce_sum(losses, name='losses_sum'),
                self.batch_size,
                name='average_cost')
            tf.summary.scalar('cost', self.cost)

# =============================================================================
    
    def ms_error(self, labels, logits):
        return tf.square(tf.subtract(labels,logits))

    def _weight_variable(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1.,)
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)
    
if __name__ == '__main__':
    # 搭建 LSTMRNN 模型
    model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE)
    sess = tf.Session()
   
    sess.run(tf.global_variables_initializer())
    
    plt.figure(1)
    plt.ion()   # 设置连续 plot
    plt.show()
    
    m1 = []
    m2 = []
    # 训练 200 次
    for step in range(5000):
        seq, res, xs = get_batch(step)  # 提取 batch data
       
        if step == 0:
        # 初始化 data
            feed_dict = {
                    model.xs: seq,
                    model.ys: res,
            }
        else:
            feed_dict = {
                model.xs: seq,
                model.ys: res,
                model.cell_init_state: state    # 保持 state 的连续性
            }
        
        # 训练
        _, cost, state, pred = sess.run(
            [model.train_op, model.cost, model.cell_final_state, model.pred],
            feed_dict=feed_dict)
      
        
        plt.plot(xs[0, :], scaler2.inverse_transform(res[0].flatten().reshape(-1,125)).reshape(125,-1), 'r', 
                 xs[0, :], scaler2.inverse_transform(pred.flatten()[:TIME_STEPS].reshape(-1,125)).reshape(125,-1), 'b--')
     
        plt.draw()
        plt.pause(0.3)  # 每 0.3 s 刷新一次
        m1.append(scaler2.inverse_transform(pred.flatten()[:TIME_STEPS].reshape(-1,125)).tolist())
        m2.append(scaler2.inverse_transform(res[0].flatten().reshape(-1,125)).tolist())
    scipy.io.savemat('RNN_Train.mat', {'Pred':np.array(m1).squeeze().flatten(), 'Truth':np.array(m2).squeeze().flatten()})
        
# =============================================================================
    print("......Test Start......")
    result = np.zeros((1000,4))
    error = []
    plt.ion()   # 设置连续 plot
   
    x = []
    y0 = []
    y1 = []
    y2 = []
    y3 = []
#    y4 = []
#    y5 = []
#    y6 = []
#    y7 = []
#    y8 = []
    
    for i in range(1000):
        x.append(i)
        seq = dataPPGtes[i:i+1,:]
        seq = seq[:, :, np.newaxis]
        res = dataABPtes[i:i+1,:]
        res = res[:, :, np.newaxis]
        feed_dict = {
                        model.xs: seq,
                        model.ys: res,      
                }
        state, pred = sess.run(
                [model.cell_final_state, model.pred], feed_dict=feed_dict)
        
        err = scaler4.inverse_transform(pred.flatten()[:TIME_STEPS].reshape(-1,125)).reshape(125,1) - scaler4.inverse_transform(res[0].flatten().reshape(-1,125)).reshape(125,-1)
        error.append(err.tolist())
        AbpMae = tf.reduce_mean(tf.abs(err))
        result[i,0] = sess.run(AbpMae)
#        print("ABP_MAE: %.2f" % result[i,0])
        y0.append(result[i,0])
        
        AbpRmse = tf.sqrt(tf.reduce_mean((err)**2))
        result[i,1] = sess.run(AbpRmse)
#        print("ABP_RMSE: %.2f" % result[i,1])
        y1.append(result[i,1])
        
        AbpRsqr = r2_score(scaler2.inverse_transform(pred.flatten()[:TIME_STEPS].reshape(-1,125)).reshape(125,1), scaler2.inverse_transform(res[0].flatten().reshape(-1,125)).reshape(125,-1))
        result[i,2] = AbpRsqr
#        print("ABP_R Squared: %.2f" % result[i,2])
        y2.append(result[i,2])
        
        AbpStd = np.std(err)
        result[i,3] = AbpStd
#        print("ABP_STD: %.2f" % result[i,3])
        y3.append(result[i,3])
#        plt.plot(x, y0,'r', x, y1,'g', x, y2,'b')
#        plt.plot(x, y3,'r', x, y4,'g', x, y5,'b')
#        plt.plot(x, y6,'r', x, y7,'g', x, y8,'b')
        plt.figure(2)
        plt.plot(x, y0, 'b')
        
        plt.figure(3)
        plt.plot(x, y1, 'b')
        
        plt.figure(4)
        plt.plot(x, y2, 'b')
        
        plt.figure(5)
        plt.plot(x, y3, 'b')
        
        plt.draw()
        plt.pause(0.1)
        plt.show()
		
    print("ABP_MAE Avr = %.2f" % sess.run(tf.reduce_mean(y0)))
#    print("SBP_MAE Avr = %.2f" % sess.run(tf.reduce_mean(y1)))
#    print("DBP_MAE Avr = %.2f" % sess.run(tf.reduce_mean(y2)))
#    
    print("ABP_RMSE Avr = %.2f" % sess.run(tf.reduce_mean(y1)))
#    print("SBP_RMSE Avr = %.2f" % sess.run(tf.reduce_mean(y4)))
#    print("DBP_RMSE Avr = %.2f" % sess.run(tf.reduce_mean(y5)))
    
    print("ABP_R Sqr Avr = %.2f" % sess.run(tf.reduce_mean(y2)))
#    print("SBP_R Sqr Avr = %.2f" % sess.run(tf.reduce_mean(y7)))
#    print("DBP_R Sqr Avr = %.2f" % sess.run(tf.reduce_mean(y8)))
#    
    print("ABP_STD Avr = %.2f" % sess.run(tf.reduce_mean(y3)))
    scipy.io.savemat('RNN_Test.mat',{'MAE':result[:, 0], 'RMSE':result[:, 1], 'R2Sq':result[:, 2], 'STD':result[:, 3],
                                   'Error':np.array(error).squeeze().flatten()})
