from rpn_net import generate_anchor

import numpy as np
np.set_printoptions(threshold=np.inf)
def generate_all_anchors(width = 37,height = 50,feat_stride = 16):
    base_anchors = generate_anchor.generate_base_anchor(x_ctr=0,y_ctr=0)
    x_ctr = np.arange(width)*feat_stride
    y_ctr = np.arange(height)*feat_stride
    [X, Y] = np.meshgrid(x_ctr, y_ctr)
    ctrs = np.vstack((X.ravel(),Y.ravel(),X.ravel(),Y.ravel())).transpose()
    all_anchors = []
    for ctr in ctrs:
        x1 = ctr[0]+base_anchors[:,0]
        y1 = ctr[1]+base_anchors[:,1]
        x2 = ctr[2]+base_anchors[:,2]
        y2 = ctr[3]+base_anchors[:,3]
        all_anchors.append(np.array([x1,y1,x2,y2]).transpose())
    all_anchors = np.array(all_anchors)
    all_anchors = all_anchors.reshape([-1,4])
    return all_anchors
if __name__ == '__main__':
    all_anchors = generate_all_anchors()
    # print(all_anchors)