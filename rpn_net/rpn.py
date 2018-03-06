import tensorflow as tf
import numpy as np
# np.set_printoptions(threshold=np.nan)
from tensorflow.contrib import slim
import vgg16_net
class RPN_Net():
    def __init__(self,vgg_network,im_info):
        self.vgg_network = vgg_network
        self.im_info = im_info

    def rpn_network(self):
        endpoints = {}
        endpoints['im_info'] = self.im_info
        vgg_network = self.vgg_network

        vgg_net_shape = vgg_network.get_shape().as_list()
        W = vgg_net_shape[1]
        H = vgg_net_shape[2]
        with slim.arg_scope([slim.conv2d],weights_initializer=tf.truncated_normal_initializer(0.0, stddev=0.01),
                                          biases_initializer=tf.constant_initializer(0.0),
                                          kernel_size=3,
                                          activation_fn=tf.nn.relu,
                                          stride=1,
                                          padding='VALID'):
            rpn_conv_net = slim.conv2d(inputs=vgg_network,num_outputs=256,padding='SAME')
            rpn_cls_score = slim.conv2d(inputs=rpn_conv_net,
                                        num_outputs=18,
                                        kernel_size=1,
                                        activation_fn=None,
                                        scope='rpn_cls_score')
            rpn_cls_score_reshape = tf.reshape(rpn_cls_score,[-1,W,2*H,9])
            rpn_cls_prob = slim.softmax(rpn_cls_score_reshape)
            rpn_cls_prob = tf.reshape(rpn_cls_prob,[-1,W,H,18],name='rpn_cls_prob')
            endpoints['rpn_cls_prob'] = rpn_cls_prob

            rpn_bbox_regression = slim.conv2d(inputs=rpn_conv_net,
                                              num_outputs=36,
                                              kernel_size=1,
                                              activation_fn=None,
                                              scope='rpn_bbox_regression')

            endpoints['rpn_bbox_regression'] = rpn_bbox_regression

        return endpoints

if __name__ == '__main__':
    vgg16_network = vgg16_net.VGG16_Net()
    inputs = tf.placeholder(tf.float16, shape=[1, 800, 600, 3], name='inputs')
    im_info = tf.placeholder(dtype='int32', shape=[1,2], name='im_info')

    network = vgg16_network.network(inputs=inputs)
    rpn_network = RPN_Net(network,im_info=im_info).rpn_network()
    print(rpn_network)
    M = 1000
    N = 600
    with tf.Session() as sess:
        x = np.ones([1, 800, 600, 3], dtype='float16')

        sess.run(tf.global_variables_initializer())
        sess.run(rpn_network, {inputs: x, im_info: np.reshape([M,N],newshape=[1,2])})
        # print(sess.run(rpn_network, {inputs:x,im_info:[16]}))


