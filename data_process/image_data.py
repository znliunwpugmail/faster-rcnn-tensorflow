from data_process import pascal_voc_test

def images_info(xmls_path,images_path):
    pascal_process = pascal_voc_test.Pascal_Process(xmls_path=xmls_path,images_path=images_path)
    images = pascal_process.pascal_image_read()
    return images
if __name__ == '__main__':
    xmls_path = 'E:/VOC2007-original/Annotations'
    images_path = 'E:/VOC2007-original/JPEGImages'
    images_info(xmls_path=xmls_path,images_path=images_path)