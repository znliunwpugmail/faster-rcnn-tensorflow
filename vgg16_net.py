import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np

class VGG16_Net():
    def __init__(self):
        pass

    def network(self,inputs):
        with slim.arg_scope([slim.conv2d],
                            weights_initializer=tf.truncated_normal_initializer(0.0, stddev=0.01),
                            biases_initializer=tf.constant_initializer(0.0),
                            # weights_regularizer=slim.l2_regularizer(0.0005),
                            kernel_size=3,
                            activation_fn=tf.nn.relu,
                            padding='SAME'):
            with slim.arg_scope([slim.max_pool2d],kernel_size=2, stride = 2,padding='VALID'):
                with tf.variable_scope('block1'):
                    net = slim.repeat(inputs=inputs,repetitions=2,layer=slim.conv2d,num_outputs=64,scope='conv1')
                    net = slim.max_pool2d(inputs=net,scope='pool_1')
                with tf.variable_scope(name_or_scope='block2'):
                    net = slim.repeat(inputs=net,repetitions=2,layer=slim.conv2d,num_outputs=128,scope='conv2')
                    net = slim.max_pool2d(inputs=net,scope='pool_2')
                with tf.variable_scope(name_or_scope='block3'):
                    net = slim.repeat(inputs=net,repetitions=3,layer=slim.conv2d,num_outputs=256,scope='conv3')
                    net = slim.max_pool2d(inputs=net,scope='pool_3')
                with tf.variable_scope(name_or_scope='block4'):
                    net = slim.repeat(inputs=net,repetitions=3,layer=slim.conv2d,num_outputs=512,scope='conv4')
                    net = slim.max_pool2d(inputs=net,scope='pool_4')
                with tf.variable_scope(name_or_scope='block5'):
                    net = slim.repeat(inputs=net,repetitions=3,layer=slim.conv2d,num_outputs=512,scope='conv5')
        return net

if __name__ == '__main__':
    vgg16_net = VGG16_Net()
    x = np.ones([1,800,600,3],dtype='float32')
    inputs = tf.placeholder(tf.float32,shape=[1,800,600,3],name='inputs')
    network = vgg16_net.network(inputs=inputs)
    print(network)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(network,{inputs:x})

