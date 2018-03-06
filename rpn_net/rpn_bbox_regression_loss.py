import tensorflow as tf
import numpy as np
import vgg16_net
from data_process import target_data
from rpn_net import rpn
def rpn_bbox_reg_loss():
    pass

if __name__ == '__main__':
    vgg16_network = vgg16_net.VGG16_Net()
    inputs = tf.placeholder(tf.float16, shape=[1, 800, 600, 3], name='inputs')
    im_info = tf.placeholder(dtype='int32', shape=[1,2], name='im_info')

    network = vgg16_network.network(inputs=inputs)
    rpn_network = rpn.RPN_Net(network,im_info=im_info).rpn_network()
    print(rpn_network)
    rpn_bbox_regression = rpn_network['rpn_bbox_regression']
    rpn_cls_prob = rpn_network['rpn_cls_prob']
    print(rpn_bbox_regression)
    print(rpn_cls_prob)
    M = 1000
    N = 600
    with tf.Session() as sess:
        x = np.ones([1, 800, 600, 3], dtype='float16')

        sess.run(tf.global_variables_initializer())
        print(sess.run(rpn_network, {inputs: x, im_info: np.reshape([M,N],newshape=[1,2])}))
        # print(sess.run(rpn_network, {inputs:x,im_info:[16]}))

