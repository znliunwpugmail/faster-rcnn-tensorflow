import tensorflow as tf
import os
zero_out_module = tf.load_op_library('./zero_out.so')
with tf.Session('') as sess:
    x = zero_out_module.zero_out([[100, 2], [3, 4]]).eval()

print(x)