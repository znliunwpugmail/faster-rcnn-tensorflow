import numpy as np

def nms(dets,thresh):
    xmin = dets[:,0]
    ymin = dets[:,1]
    xmax = dets[:,2]
    ymax = dets[:,3]
    confidence = dets[:,4]
    areas = np.maximum(0,ymax-ymin+1)*np.maximum(0,xmax-xmin+1)
    confidence_order = np.argsort(confidence)[::-1]
    r_rect = []
    while len(confidence_order)>0:
        r_rect.append(confidence_order[0])
        x1 = np.maximum(dets[confidence_order[0],0],dets[confidence_order[1:],0])
        y1 = np.maximum(dets[confidence_order[0], 1], dets[confidence_order[1:], 1])
        x2 = np.minimum(dets[confidence_order[0], 2], dets[confidence_order[1:], 2])
        y2 = np.minimum(dets[confidence_order[0], 2], dets[confidence_order[1:], 2])
        area = np.maximum(x2-x1+1,0)*np.maximum(y2-y1+1,0)
        area1 = areas[confidence_order[0]]
        area2 = areas[confidence_order[1:]]
        iou = area/(area1+area2-area)
        inds = np.where(iou<thresh)
        confidence_order = confidence_order[inds[0]+1]
    return r_rect

if __name__ == '__main__':
    dets = np.array([
                    [204, 102, 358, 250, 0.5],
                    [257, 118, 380, 250, 0.7],
                    [280, 135, 400, 250, 0.6],
                    [255, 118, 360, 235, 0.7],
                    [10, 12, 40, 56, 0.6]])
    thresh = 0.3
    rect = nms(dets,thresh)
    print(rect)