import tensorflow as tf

'''
一次取回多个tensor值
'''
t1 = tf.constant(3)
t2 = tf.constant(4)
t3 = tf.constant(5)
add_r = tf.add(t2,t3)   # 9
mul_r = tf.multiply(t1,add_r)  # 27


with tf.Session() as ss:
    result = ss.run([add_r,mul_r])
    print(result,type(result),sep="\n")  #[9, 27]    <class 'list'>

