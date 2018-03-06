import tensorflow as tf
import numpy as np
import vgg16_net
from data_process import target_data
from rpn_net import rpn

def rpn_bbox_regression_loss(rpn_bbox_regression,
                             target_data,
                             width = 37,
                             height = 50,
                             num_base_anchors = 9,
                             batch_size = 1):
    target_bboxes = target_datas[1]
    target_labels_inside_weight = target_datas[2]
    target_labels_outside_weight = target_datas[3]
    # target_labels = np.transpose(np.reshape(target_labels,
    #                                         newshape=[batch_size,width,height,
    #                                                   num_base_anchors]),
    #                              (0,2,1,3))
    target_bboxes = np.transpose(np.reshape(target_bboxes,
                                            newshape=[batch_size,width,height,
                                                      num_base_anchors*4]),
                                 (0,2,1,3))
    target_labels_inside_weight = np.transpose(np.reshape(target_labels_inside_weight,
                                            newshape=[batch_size, width, height,
                                                      num_base_anchors*4]),
                                 (0, 2, 1, 3))
    target_labels_outside_weight = np.transpose(np.reshape(target_labels_outside_weight,
                                                          newshape=[batch_size, width,
                                                                    height, num_base_anchors*4]),
                                               (0, 2, 1, 3))

    # print(target_bboxes.shape)
    # print(target_labels_inside_weight.shape)
    # print(target_labels_outside_weight.shape)
    target_bboxes = tf.convert_to_tensor(target_bboxes)
    target_labels_inside_weight = tf.convert_to_tensor(target_labels_inside_weight)
    target_labels_outside_weight = tf.convert_to_tensor(target_labels_outside_weight)
    smooth_loss = smooth_l1(target_bboxes=target_bboxes,
                              target_labels_inside_weight = target_labels_inside_weight,
                              target_labels_outside_weight=target_labels_outside_weight,
                              rpn_bbox_regression = rpn_bbox_regression,
                              name = 'smooth_loss',
                              sigma=3.0)
    rpn_bbox_loss = tf.reduce_mean(tf.reduce_sum(smooth_loss,
                                                 reduction_indices=[1, 2, 3]),
                                   name='rpn_bbox_loss')
    return rpn_bbox_loss

def smooth_l1(target_bboxes,
              target_labels_inside_weight,
              target_labels_outside_weight,
              rpn_bbox_regression,
              name,
              sigma = 1.0):
    """
    :param target_labels:
    :param target_bboxes:
    :param target_labels_inside_weight:
    :param sigma:
    :return:
    smooth_l1 = 0.5*(sigma*x)^2 if |x|<1/sigma^2
                |x| - 0.5/sigma^2 otherwise
                x = inside_weight*(target_bboxes - rpn_bbox_regression)
    """
    sigma_square = sigma*sigma
    x = tf.multiply(target_labels_inside_weight,
                                tf.subtract(target_bboxes,rpn_bbox_regression))
    smooth_sign = tf.cast(tf.less(tf.abs(x),1.0/sigma_square),tf.float32)
    smooth_1 = tf.multiply(tf.multiply(x,x),sigma_square*0.5)
    smooth_2 = tf.subtract(tf.abs(x),0.5/sigma_square)
    loss = tf.add(tf.multiply(smooth_1,smooth_sign),tf.multiply(smooth_2,tf.subtract(1.0,smooth_sign)))
    smooth_loss = tf.multiply(target_labels_outside_weight,loss,name=name)
    return smooth_loss

def rpn_cls_loss(rpn_cls_prob,target_datas):
    target_labels = target_datas[0]
    # target_labels = np.array(target_labels)
    target_labels = target_labels.astype(np.int32)
    print(type(target_labels))

    # print(target_labels.dtype)
    target_labels = tf.convert_to_tensor(target_labels,name='target_labels')
    # print(target_labels)

    rpn_select = tf.where(tf.not_equal(target_labels,-1))
    rpn_cls_score = tf.gather(rpn_cls_prob,rpn_select)
    rpn_cls_score = tf.reshape(rpn_cls_score,[-1,2],name='rpn_cls_score')
    rpn_labels = tf.gather(target_labels,rpn_select)
    rpn_labels = tf.reshape(rpn_labels,[-1],name='rpn_labels')
    # rpn_labels = tf.cast(x=rpn_labels,dtype=tf.float32)
    # print(rpn_labels)
    # print(rpn_cls_score)
    rpn_cross_entropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=rpn_cls_score,
                                                       labels=rpn_labels),
        name='rpn_cls_loss')
    # print(rpn_cross_entropy)
    return rpn_cross_entropy


if __name__ == '__main__':
    target_datas = target_data.generate_target_datas()
    vgg16_network = vgg16_net.VGG16_Net()
    inputs = tf.placeholder(tf.float32, shape=[1, 800, 600, 3], name='inputs')
    im_info = tf.placeholder(dtype='int32', shape=[1,2], name='im_info')

    network = vgg16_network.network(inputs=inputs)
    rpn_network = rpn.RPN_Net(network,im_info=im_info).rpn_network()
    rpn_bbox_regression = rpn_network['rpn_bbox_regression']
    print(rpn_bbox_regression)
    rpn_cls_prob = rpn_network['rpn_cls_prob']
    # print(rpn_cls_prob)
    M = 1000
    N = 600
    rpn_cls_prob_loss = rpn_cls_loss(rpn_cls_prob,target_datas)
    print(rpn_cls_prob_loss)
    rpn_bbox_loss = rpn_bbox_regression_loss(rpn_bbox_regression,target_datas)
    print(rpn_bbox_loss)
    # with tf.Session() as sess:
    #     x = np.ones([1, 800, 600, 3], dtype='float16')
    #
    #     sess.run(tf.global_variables_initializer())
    #     print(sess.run(rpn_network, {inputs: x, im_info: np.reshape([M,N],newshape=[1,2])}))
    #     # print(sess.run(rpn_network, {inputs:x,im_info:[16]}))

