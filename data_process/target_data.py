import numpy as np
from numpy import random
from rpn_net import generate_all_anchors
from data_process import roi_data
from rpn_net import bbox_transform

def get_inds_inside(all_anchors,tolerate_border = 0,im_info=[600,800,3]):
    # print(all_anchors.shape)
    inds_inside = np.where(
        (all_anchors[:, 0] >= -tolerate_border) &
        (all_anchors[:, 1] >= -tolerate_border) &
        (all_anchors[:, 2] < im_info[1] + tolerate_border) &  # width
        (all_anchors[:, 3] < im_info[0] + tolerate_border)  # height
    )[0]

    return inds_inside

def IoUs_compute(inside_anchors,gt_bboxes):
    num_anchors = inside_anchors.shape[0]
    num_bboxes = gt_bboxes.shape[0]
    IoUs = np.zeros([num_anchors,num_bboxes],dtype=np.float32)

    area_anchors = (inside_anchors[:,2]-inside_anchors[:,0]+1)\
                   *(inside_anchors[:,3]-inside_anchors[:,1]+1)
    area_bboxes = (gt_bboxes[:,2]-gt_bboxes[:,0]+1)\
                  *(gt_bboxes[:,3]-gt_bboxes[:,1]+1)
    for i in range(num_anchors):
        for j in range(num_bboxes):
            x1 = min(inside_anchors[i,2],gt_bboxes[j,2])
            x2 = max(inside_anchors[i,0],gt_bboxes[j,0])
            y1 = min(inside_anchors[i,3],gt_bboxes[j,3])
            y2 = max(inside_anchors[i,1],gt_bboxes[j,1])
            if x1<x2 or y1<y2:
                area_IoU = 0
            else:
                area_IoU = (x1-x2+1)*(y1-y2+1)

            # print(area_IoU)
            IoUs[i,j] = area_IoU/(area_anchors[i]+area_bboxes[j]-area_IoU)
    return IoUs
def generate_target_data(all_anchors,gt_bboxes,fg_thresh = 0.7,bg_thresh = 0.3,
                  num_fg = 128,num_labels = 256):
    inds_inside = get_inds_inside(all_anchors)
    inside_anchors = all_anchors[inds_inside,:]
    target_labels = np.empty([len(inds_inside),],dtype=np.float32)
    target_labels.fill(-1)
    IoUs = IoUs_compute(inside_anchors,gt_bboxes)

    argmax_IoU = IoUs.argmax(1)
    max_IoU = IoUs[np.arange(len(inds_inside)),argmax_IoU]

    gt_argmax_IoU = IoUs.argmax(0)
    gt_max_IoU = IoUs[gt_argmax_IoU,np.arange(IoUs.shape[1])]

    gt_argmax_IoU = np.where(IoUs==gt_max_IoU)[0]
    target_labels[gt_argmax_IoU]=1
    target_labels[max_IoU<bg_thresh]=0
    target_labels[max_IoU>fg_thresh]=1
    fg_inds = np.where(target_labels==1)[0]

    if len(fg_inds)>num_fg:
        clip_inds = random.choice(
            fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        target_labels[clip_inds] = -1

    num_bg = num_labels-len(np.where(target_labels==1)[0])
    bg_inds = np.where(target_labels==0)[0]
    if len(bg_inds)>num_bg:
        clip_inds = random.choice(
            bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        target_labels[clip_inds] = -1
    # print(IoUs)
    # argmax_overlaps = overlaps.argmax(axis=1)
    # max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
    # gt_argmax_overlaps = overlaps.argmax(axis=0)
    # gt_max_overlaps = overlaps[gt_argmax_overlaps,
    #                            np.arange(overlaps.shape[1])]
    # gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]
    # print(target_labels.shape)
    # print(gt_bboxes[argmax_IoU,:].shape)

    target_bboxes_tuple = bbox_transform.bbox_transform_target(
                                       inside_anchors,gt_bboxes[argmax_IoU,:])
    target_bboxes = []
    target_bboxes.append([target_bboxes_tuple.targets_dx,
                          target_bboxes_tuple.targets_dy,
                           target_bboxes_tuple.targets_dw,
                           target_bboxes_tuple.targets_dh])
    target_bboxes = np.array(target_bboxes).transpose()
    target_bboxes = target_bboxes.squeeze()
    # print(target_bboxes)
    # print(target_bboxes.shape)

    target_labels_inside_weight = np.zeros(shape=[len(inds_inside),4],dtype=np.float32)
    target_labels_inside_weight[np.where(target_labels==1)[0],:]= 1

    target_labels_outside_weight = np.zeros(shape=[len(inds_inside),4],dtype=np.float32)
    num_examples = np.sum(target_labels>=0)
    negative_weights = np.ones(shape=[1,4],dtype=np.float32)*1.0/num_examples
    positive_weights = np.ones(shape=[1,4],dtype=np.float32)*1.0/num_examples
    target_labels_outside_weight[target_labels==0,:] = negative_weights
    target_labels_outside_weight[target_labels==1,:] = positive_weights
    target_labels = map_to_origin(data=target_labels,
                                  inds_inside=inds_inside,
                                  num = len(all_anchors),
                                  default_data=-1)
    target_bboxes = map_to_origin(data=target_bboxes,
                                   inds_inside=inds_inside,
                                   num = len(all_anchors),
                                   default_data=0)
    target_labels_inside_weight = map_to_origin(data=target_labels_inside_weight,
                                   inds_inside=inds_inside,
                                   num = len(all_anchors),
                                   default_data=0)
    target_labels_outside_weight = map_to_origin(data=target_labels_outside_weight,
                                                 inds_inside=inds_inside,
                                                 num = len(all_anchors),
                                                 default_data=0)
    # print(target_bboxes)
    # print(target_labels)
    # print(np.where(target_labels==1))
    # print(np.where(target_labels==0))
    # print(len(np.where(target_labels==-1)[0]))

    return target_labels,target_bboxes,target_labels_inside_weight,target_labels_outside_weight

def map_to_origin(data,num,inds_inside,default_data=-1):
    ret_shape = np.array(data.shape)
    if len(ret_shape)>=1:
        ret_shape[0] = num
    ret = np.empty(ret_shape,dtype=np.float32)
    ret.fill(default_data)
    if len(ret_shape)==1:
        ret[inds_inside] = data
    elif len(ret_shape)>1:
        ret[inds_inside,:] = data
    return ret
def generate_target_datas(width = 600,height = 800):
    all_anchors = generate_all_anchors.generate_all_anchors()
    images_info, gt_bboxes, gt_labels = roi_data.xmls_info()
    for i in range(len(gt_bboxes)):
        image_id = gt_bboxes[i, 0]
        image_width = images_info[image_id, 0]
        image_height = images_info[image_id, 1]
        # print(images_info[image_id,0:2])
        # print(gt_bboxes[i, :])
        gt_bboxes[i, 1:5:2] = gt_bboxes[i, 1:5:2] / image_width * width
        gt_bboxes[i, 2:5:2] = gt_bboxes[i, 2:5:2] / image_height * height
        # print(gt_bboxes[i,:])
    print('gt_bboxes prepared end')
    target_data = generate_target_data(all_anchors, gt_bboxes=gt_bboxes[0:2, 1:5])
    return target_data

if __name__ == '__main__':
    # width = 600
    # height = 800
    # all_anchors = generate_all_anchors.generate_all_anchors()
    # images_info, gt_bboxes, gt_labels = roi_data.xmls_info()
    # for i in range(len(gt_bboxes)):
    #     image_id = gt_bboxes[i,0]
    #     image_width = images_info[image_id,0]
    #     image_height = images_info[image_id,1]
    #     # print(images_info[image_id,0:2])
    #     # print(gt_bboxes[i, :])
    #     gt_bboxes[i,1:5:2]=gt_bboxes[i,1:5:2]/image_width*width
    #     gt_bboxes[i,2:5:2]=gt_bboxes[i,2:5:2]/image_height*height
    #     # print(gt_bboxes[i,:])
    # print('gt_bboxes prepared end')
    # target_datas = generate_target_data(all_anchors,gt_bboxes=gt_bboxes[0:2,1:5])
    # print(target_datas[0].shape,target_datas[1].shape,target_datas[2].shape)
    # print(target_labels.shape)

    target_datas = generate_target_datas()
    # print(target_datas[0])
