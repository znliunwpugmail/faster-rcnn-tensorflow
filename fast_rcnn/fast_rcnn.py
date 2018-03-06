import tensorflow as tf
import numpy as np
from tensorflow.contrib import slim

class fast_rcnn():
    def __init__(self):
        pass
    def fc_drop(self,network,is_training=True):
        # network = tf.reshape(network, [300 * 7 * 7 * 512])
        fast_rcnn_endpoints = []
        net = slim.fully_connected(inputs=network,
                                   num_outputs=4096,
                                   activation_fn=tf.nn.relu,
                                   weights_initializer=tf.truncated_normal_initializer,
                                   scope='fc6')
        net = slim.dropout(inputs=net,
                           keep_prob=0.5,
                           is_training=is_training,
                           scope='dropout_6')
        net = slim.fully_connected(inputs=net,
                                   num_outputs=4096,
                                   activation_fn=tf.nn.relu,
                                   weights_initializer=tf.truncated_normal_initializer,
                                   scope='fc7')
        net = slim.dropout(inputs=net,
                           keep_prob=0.5,
                           is_training=is_training,
                           scope='dropout_7')

        bbox_pred_net = slim.fully_connected(inputs=net,
                                             num_outputs=20*300,
                                             weights_initializer=tf.truncated_normal_initializer,
                                             scope='bbox_red_net_fast_rcnn')
        fast_rcnn_endpoints.append(bbox_pred_net)
        cls_prob_net = slim.fully_connected(inputs=net,
                                            num_outputs=4*300,
                                            weights_initializer=tf.truncated_normal_initializer,
                                            scope='cls_prob_net_fast_rcnn')
        fast_rcnn_endpoints.append(cls_prob_net)
        return fast_rcnn_endpoints


if __name__ == '__main__':
    rois_feature = np.ones(dtype='float32',shape=[300,7,7,512])
    inputs = tf.placeholder(tf.float32,shape=[300,7,7,512])

    fast_rcnn = fast_rcnn()
    fast_rcnn_endpoints = fast_rcnn.fc_drop(inputs)
    fast_rcnn_bbox_pred = fast_rcnn_endpoints[0]
    fast_rcnn_cls_prob = fast_rcnn_endpoints[1]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(fast_rcnn_bbox_pred,{inputs:rois_feature}))
        print(sess.run(fast_rcnn_cls_prob,{inputs:rois_feature}))


