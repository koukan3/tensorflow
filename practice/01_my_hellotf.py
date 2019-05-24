import tensorflow as tf

'''
构建一个计算图graph
'''


#创建一个运算操作的源（source op）,这也是一个节点node
matrix1 = tf.constant([[3,4]])
#第二个op,这也是一个节点node
matrix2 = tf.constant([[5],[6]])
#第三个op，这也是一个node
mul_add = tf.matmul(matrix1,matrix2)

#在会话中启动计算图.
'''
如果在Session构建中没有指定graph参数，就使用默认的图。
如果需要使用多个graph，需要为每个graph都开启一个session。
但一个graph 通过在Session构造器中传递可以实现在多个session中使用。
'''
ss = tf.Session()
fetch_matrix1 = ss.run(matrix1)
print("获取第一个op的值： ",fetch_matrix1," , type = ",type(fetch_matrix1))
#触发三个op的执行（通常是并发执行）
result = ss.run(mul_add)
print(result," , type = ",type(result))
ss.close()
'''
#任务完后，关闭session以释放资源。
1.ss.close()显示的关闭
2.with代码块，自动关闭
'''
with tf.Session() as ss:
    # 指派特定的CPU/GPU来执行
    with tf.device("/cpu:0"):
        result = ss.run([mul_add])
        print(result,type(result))



