import xml.etree.ElementTree as ET
import cv2
import os
import collections
import numpy as np

Annotations = {}

classes = ('__background__', # always index 0
                         'aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')

classes_dict = dict((zip(classes,range(len(classes)))))
xml_name = '000001.xml'

class Pascal_Process():
    def __init__(self,xmls_path,images_path,result_images_path):
        self.xmls_path = xmls_path
        self.images_path = images_path
        self.result_images_path = result_images_path
    def __init__(self,xmls_path,images_path):
        self.xmls_path = xmls_path
        self.images_path = images_path
    def pascal_xml_read(self):
        xmls_path = self.xmls_path
        images_path = self.images_path
        # result_images_path = self.result_images_path
        xml_names = os.listdir(xmls_path)
        xmls_info = []
        for xml_name in xml_names:

            image_info = collections.namedtuple('image_info',
                                                ['image_name','width', 'height', 'depth', 'objects_info'])

            image_name = xml_name.split('.')[0]+'.jpg'
            image_path = os.path.join(images_path,image_name)
            if (os.path.isfile(image_path)) == False:
                continue
            xml_path = os.path.join(xmls_path,xml_name)
            tree = ET.ElementTree(file=xml_path)
            root = tree.getroot()

            size_root = root.find('size')
            M = size_root.find('width').text
            N = size_root.find('height').text
            depth = size_root.find('depth').text
            M = int(M)
            N = int(N)
            depth = int(depth)
            # im = cv2.imread(image_path)

            objects_info = []
            for object in root.findall('object'):
                class_name = object.find('name').text
                class_id = classes_dict[class_name]
                bndbox = object.find('bndbox')
                xmin = bndbox.find('xmin').text
                xmin = int(xmin)
                ymin = bndbox.find('ymin').text
                ymin = int(ymin)
                xmax = bndbox.find('xmax').text
                xmax = int(xmax)
                ymax = bndbox.find('ymax').text
                ymax = int(ymax)
                objects_info.append([class_name,class_id,xmin,ymin,xmax,ymax])
                # cv2.rectangle(im, (xmin, ymin), (xmax, ymax), (0, 0, 255), 4)
                # font = cv2.FONT_HERSHEY_PLAIN
                # text = class_name + ':' + str(class_id)
                # cv2.putText(im, text, (xmin, ymin), font, 1.5, (0, 255, 0), 2)
            im_info=image_info(image_name=image_name,width=M,height=N,depth=depth,objects_info=objects_info)
            xmls_info.append(im_info)
            # cv2.imwrite(os.path.join(result_images_path,image_name),im)
        return xmls_info

    def pascal_image_read(self):
        xmls_path = self.xmls_path
        images_path = self.images_path
        # result_images_path = self.result_images_path
        image_names = os.listdir(images_path)
        images = []
        for image_name in image_names:

            image_info = collections.namedtuple('image_info',
                                                ['image_name', 'width', 'height', 'depth', 'objects_info'])

            xml_name = image_name.split('.')[0] + '.xml'
            xml_path = os.path.join(xmls_path, xml_name)
            if (os.path.isfile(xml_path)) == False:
                continue
            image_path = os.path.join(images_path, image_name)
            im = cv2.imread(image_path)
            images.append(im)

            # cv2.imwrite(os.path.join(result_images_path,image_name),im)
        images = np.array(images)
        return images

if __name__ == '__main__':
    pascal_process = Pascal_Process(xmls_path='E:/VOC2007-original/Annotations',
                                    images_path='E:/VOC2007-original/JPEGImages',
                                    result_images_path='E:/VOC2007-original/Liu_Path')
    xmls_info = pascal_process.pascal_xml_read()
    print(len(xmls_info))