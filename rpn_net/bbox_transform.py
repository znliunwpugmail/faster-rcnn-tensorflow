import numpy as np
import collections

def bbox_transform_target(ex_boxes,gt_boxes):
    ex_ctr_xs,ex_ctr_ys,ex_widths,ex_heights = bbox_transform_a(ex_boxes)
    gt_ctr_xs,gt_ctr_ys,gt_widths,gt_heights = bbox_transform_a(gt_boxes)
    targets = collections.namedtuple('targets',
                                     ['targets_dx','targets_dy',
                                      'targets_dw','targets_dh'])
    targets_dx = (ex_ctr_xs-gt_ctr_xs)/gt_widths
    targets_dy = (ex_ctr_ys-gt_ctr_ys)/gt_heights
    targets_dw = np.log(ex_widths/gt_widths)
    targets_dh = np.log(ex_heights/gt_heights)

    return targets(targets_dx=targets_dx,
                   targets_dy=targets_dy,
                   targets_dw=targets_dw,
                   targets_dh=targets_dh)

def target_transform_bbox(targets,ex_boxes):
    ex_ctr_xs, ex_ctr_ys, ex_widths, ex_heights = bbox_transform_a(ex_boxes)
    targets_dx = targets.targets_dx
    targets_dy = targets.targets_dy
    targets_dw = targets.targets_dw
    targets_dh = targets.targets_dh
    pred_widths = ex_widths/(np.exp(targets_dw))
    pred_heights = ex_heights/(np.exp(targets_dh))
    pred_ctr_xs = ex_ctr_xs-(targets_dx*pred_widths)
    pred_ctr_ys = ex_ctr_ys-(targets_dy*pred_heights)
    pred_x1s = pred_ctr_xs-0.5*(pred_widths-1)
    pred_x2s = pred_ctr_xs+0.5*(pred_widths-1)
    pred_y1s = pred_ctr_ys-0.5*(pred_heights-1)
    pred_y2s= pred_ctr_ys+0.5*(pred_heights-1)
    pred_boxes = np.array([pred_x1s,pred_y1s,pred_x2s,pred_y2s])
    pred_boxes = np.transpose(pred_boxes)
    return pred_boxes

def bbox_transform_a(bboxes):
    widths = bboxes[:,2]-bboxes[:,0]+1
    heights = bboxes[:, 3] - bboxes[:, 1] + 1
    ctr_xs = (bboxes[:,2]+bboxes[:,0])/2
    ctr_ys = (bboxes[:,3]+bboxes[:,1])/2
    return ctr_xs,ctr_ys,widths,heights

if __name__ == '__main__':
    ex_dets = np.array([
        [204, 102, 358, 250, 0.5],
        [257, 118, 380, 250, 0.7],
        [280, 135, 400, 250, 0.6],
        [255, 118, 360, 235, 0.7],
        [10, 12, 40, 56, 0.6]])

    gt_dets = np.array([
        [200, 100, 360, 230, 0.5],
        [207, 108, 390, 280, 0.7],
        [240, 145, 420, 250, 0.6],
        [295, 108, 340, 255, 0.7],
        [12, 10, 50, 67, 0.6]])

    targets = bbox_transform_target(ex_dets[:,0:4],gt_dets[:,0:4])
    print(targets)
    pred_boxes = target_transform_bbox(targets,ex_dets[:,:4])
    print(pred_boxes)
