from data_process import pascal_voc_test
import numpy as np
def xmls_info(xmls_path='E:/VOC2007-original/Annotations',
                       images_path='E:/VOC2007-original/JPEGImages'):
    pascal_process = pascal_voc_test.Pascal_Process(xmls_path=xmls_path,images_path=images_path)
    xmls_info = pascal_process.pascal_xml_read()
    images_info = []
    gt_bboxes = []
    gt_labels = []
    i = 0
    for xml_info in xmls_info:
        images_info.append([xml_info.width,xml_info.height,xml_info.depth])
        for object_info in xml_info.objects_info:
            gt_bboxes.append([i,object_info[2],object_info[3],object_info[4],object_info[5]])
            gt_labels.append(object_info[1])
        i = i+1
    images_info = np.array(images_info)

    gt_bboxes=np.array(gt_bboxes)
    gt_labels=np.array(gt_labels)
    return images_info,gt_bboxes,gt_labels
if __name__ == '__main__':
    xmls_path = 'E:/VOC2007-original/Annotations'
    images_path = 'E:/VOC2007-original/JPEGImages'
    xmls_info(xmls_path=xmls_path,images_path=images_path)