import numpy as np
import math
def generate_base_anchor(base_size=16,ratios=[0.5,1,2],scales=[8,16,32],x_ctr = None,y_ctr = None):
    if x_ctr is None and y_ctr is None:
        x_ctr = (base_size - 1) / 2
        y_ctr = (base_size - 1) / 2
    base_ws = np.sqrt(ratios)*16
    base_hs = np.copy(base_ws)
    base_hs = base_hs[::-1]
    base_ws[0] = math.modf(base_ws[0])[1]+1
    base_hs[0] = math.modf(base_hs[0])[1]+1
    base_ws[2] = math.modf(base_ws[2])[1]
    base_hs[2] = math.modf(base_hs[2])[1]
    scales_wh = []
    base_wh = []
    for i in range(len(base_ws)):
        base_wh.append([base_ws[i],base_hs[i]])
    base_wh = np.array(base_wh)
    for scale in scales:
        scales_wh.append(base_wh*scale)
    scales_wh = np.array(scales_wh)

    anchors = []
    for i in range(len(scales)):
        for j in range(len(ratios)):
            x1 = math.modf(x_ctr - scales_wh[i,j,1]/2)[1]
            y1 = math.modf(y_ctr - scales_wh[i,j,0]/2)[1]
            x2 = math.modf(x_ctr + scales_wh[i, j, 1]/2)[1]
            y2 = math.modf(y_ctr + scales_wh[i, j, 0]/2)[1]
            anchors.append([x1,y1,x2,y2])
    anchors = np.array(anchors)
    return anchors
if __name__ == '__main__':
    import time
    time1 = time.time()
    anchors = generate_base_anchor()
    time2 = time.time()
    print(time2-time1)
    print(anchors)