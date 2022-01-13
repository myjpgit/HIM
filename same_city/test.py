

import numpy as np
import tensorflow as tf

a=[[[1,2,3],[1,2,3]],[[4,5,6],[4,5,6]]]
c=tf.Variable([[[1],[2],[3]]],dtype=tf.float32)
c=tf.tile(c,[2,1,1])
a=tf.Variable(a,name='a',dtype=tf.float32)
b=tf.ones([1],name='b',dtype=tf.float32)
e=tf.matmul(a,c)+b
sess=tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(e))
sess.close()