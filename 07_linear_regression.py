import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing

# 立刻下载数据集
housing = fetch_california_housing(data_home="E:/MLdata/scikit_learn_data", download_if_missing=True)
# 获得X数据行数和列数
#print(type(housing))  # <class 'sklearn.datasets.base.Bunch'>
m, n = housing.data.shape
print("shape: ",m, n)   # shape:  20640 8
print(housing.data, housing.target)
print(housing.feature_names)

# 这里添加一个额外的bias输入特征(x0=1)到所有的训练数据上面，因为使用的numpy所以会立即执行
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]
# 创建两个TensorFlow常量节点X和y，去持有数据和标签
X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name='X')
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')
# 使用一些TensorFlow框架提供的矩阵操作去求theta
XT = tf.transpose(X)
# 解析解一步计算出最优解 theta =( 1/(XT*X) )*XT*y
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)
print(theta)

with tf.Session() as sess:
    theta_value = theta.eval()  # sess.run(theta)

print(theta_value)
'''
[[-3.7185181e+01]
 [ 4.3633747e-01]
 [ 9.3952334e-03]
 [-1.0711310e-01]
 [ 6.4479220e-01]
 [-4.0338000e-06]
 [-3.7813708e-03]
 [-4.2348403e-01]
 [-4.3721911e-01]]
'''

