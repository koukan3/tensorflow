import tensorflow as tf

'''
创建交互式环境和会话。
Session  》》》 InteractiveSession
session.run() 》》》tensor.eval()   operation.run()
'''
sess = tf.InteractiveSession()

m1 = tf.constant([10,20])
m2 = tf.Variable([4,8])
'''
变量的生命周期：在run方法执行时，才赋值；随着session结束而结束
'''
print("variable m2 = ",m2,type(m2)) #tensorflow.python.ops.variables.Variable
print("constant m1 = ",m1,type(m1)) #tensorflow.python.framework.ops.Tensor
##变量初始化, 否则报错：Attempting to use uninitialized value Variable
m2.initializer.run()
sub = tf.subtract(m1,m2)
print("eval: ",sub.eval())