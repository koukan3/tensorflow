import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing(data_home="d://datas/scikit_learn_data", download_if_missing=True)
m,n = housing.data.shape
print(m,n)
#抽出X，y
X = housing.data
#X = np.c_[np.ones((m,1)),housing.data]
y = housing.target
# print(y.shape) #(20640,)
#特征归一化

scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)
X = tf.constant(np.c_[np.ones((m,1)),X_scaled],dtype=tf.float32,name="X")
#X = tf.constant(X_scaled,dtype=tf.float32,name="X")
y = tf.constant(np.reshape(y,(-1,1)),dtype=tf.float32,name="y")
#计算mse
thetas = tf.Variable(tf.random_uniform(shape=((n+1,1)),minval=-1,maxval=1),name="thetas")
y_hat = tf.matmul(X,thetas)
error = y - y_hat
mse = tf.reduce_mean(tf.square(error))
#使用tf封装的方法求解梯度
gradients = tf.gradients(mse,[thetas])[0]
#计算theta值
learning_rate = 0.01
epochs = 10000
theta_op = tf.assign(thetas,thetas-learning_rate*gradients)


#session
init = tf.global_variables_initializer()

with tf.Session() as ss:
    ss.run(init)

    for i in range(epochs):
        ss.run(theta_op)
        new_thetas = thetas.eval()
        if(i%100==0):
            print("i = ",i , "mse = ",mse.eval())

    print("best thetas = ",thetas.eval())