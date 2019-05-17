import tensorflow as tf

'''
feed: 使用一个 tensor 值临时替换一个操作的输出结果. 
使用 tf.placeholder() 为这些操作创建占位符.
'''
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

t = tf.multiply(input1,input2)

with tf.Session() as ss:
    #r = ss.run(t,feed_dict={input1:[10],input2:[30]})
    #print(r)
    r = t.eval(feed_dict={input1:[10],input2:[30]})
    print(r,type(r))