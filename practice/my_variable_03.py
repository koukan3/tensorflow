import tensorflow as tf

'''
变量：维护计算图执行过程中的状态信息。例如：实现一个计数器。
'''
v_state = tf.Variable(0,name="counter")
one = tf.constant(1)
new_state = tf.add(v_state,one)
print(type(new_state))
#在调用 run() 执行表达式之前, 不会真正执行赋值操作.
update = tf.assign(v_state,new_state)
print(type(update))

#初始化变量
init_op = tf.initialize_all_variables()

with tf.Session() as ss:
    ss.run(init_op)
    print(ss.run(v_state))
    for i in range(5):
        ss.run(update)
        print("#",ss.run(v_state))