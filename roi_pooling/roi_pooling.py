import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import cv2
np.set_printoptions(threshold=np.inf)
def roi_pooling(featuremap,rois,pooling_size=7,name = 'roi'):
    with tf.variable_scope(name) as scope:
      batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
      x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1")
      y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1")
      x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2")
      y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2")
      # Won't be back-propagated to rois anyway, but to save time
      bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
      pre_pool_size = pooling_size * 2
      crops = tf.image.crop_and_resize(featuremap, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size], name="crops")

    return bboxes,slim.max_pool2d(crops, [2, 2], padding='SAME')


if __name__ == '__main__':
    image1 = cv2.imread('000001.jpg')
    image1_height = image1.shape[0]
    image1_width = image1.shape[1]
    image2 = cv2.imread('000002.jpg')
    image2_height = image2.shape[0]
    image2_width = image2.shape[1]

    image_info = np.array([[image1_height, image1_width],
                           [image1_height, image1_width]])
    input = tf.placeholder(tf.float32,shape=[2,800,600,3])
    net = slim.conv2d(inputs=input,num_outputs=64,kernel_size=3,padding='SAME',activation_fn=tf.nn.relu)
    net = slim.max_pool2d(inputs=net,kernel_size=2,stride=2)
    net = slim.conv2d(inputs = net ,num_outputs=128,kernel_size=3,padding='SAME',activation_fn=tf.nn.relu)
    net = slim.max_pool2d(inputs=net,kernel_size=2,stride=2)
    net = slim.conv2d(inputs=net,num_outputs=256,kernel_size=3,padding='SAME',activation_fn=tf.nn.relu)
    net = slim.max_pool2d(inputs=net,kernel_size=2,stride=2)
    net = slim.conv2d(inputs=net,num_outputs=512,kernel_size=3,padding='SAME',activation_fn=tf.nn.relu)
    net = slim.max_pool2d(inputs=net,kernel_size=2,stride=2)
    image1 = cv2.resize(image1,dsize=(600,800))
    image2 = cv2.resize(image2, dsize=(600, 800))
    # cv2.imshow('image1',image1)
    # cv2.waitKey()
    images = []
    images.append(image1)
    images.append(image2)
    images = np.array(images)
    images = np.reshape(images, [2,800,600,3])
    images = images.astype(dtype='float32')

    origin_dets = np.array([
        [0,48/image1_width, 240/image1_height, 195/image1_width, 371/image1_height],
        [0,8/image1_width, 12/image1_height, 352/image1_width, 498/image1_height],
        [1, 48/image1_width, 240/image1_height, 195/image1_width, 371/image1_height]])
    origin_dets = origin_dets.astype('float32')
    with tf.Session() as sess:
        images_tensor = tf.convert_to_tensor(images)
        origin_dets = tf.convert_to_tensor(origin_dets)

        print(roi_pooling(images_tensor, origin_dets))
        print(roi_pooling(net,origin_dets))
        sess.run(tf.global_variables_initializer())
        # sess.run(net,feed_dict={input:images})
        print(sess.run(roi_pooling(net,origin_dets),feed_dict={input:images}))
        # print(sess.run(roi_pooling(images_tensor,origin_dets)))
