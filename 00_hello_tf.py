import tensorflow as tf

a = tf.constant(5)
b = tf.constant(10)

c = a + b

with tf.Session() as sess:
    result = sess.run(c)
    print(result)



