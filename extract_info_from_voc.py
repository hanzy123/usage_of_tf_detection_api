## 提取person类人的相关信息
import os
import pickle

# img_dir 指的是voc数据集中存放图片的文件夹
img_dir = '/IMAGE/DIR/PATH/'
info_list = []

def deal_xml(image_list):
    for image_name in image_list:
        dict = {}
        # 去标注文件夹下获取图片对应的标注文件
        xml_path = os.path.join(root_path,'train/Annotations',image_name + '.xml')
        # train 和 test 分开制作
        # xml_path = os.path.join(root_path,'test/Annotations',image_name + '.xml')
        tree = ET.parse(xml_path)    
        root = tree.getroot() 
        filename = root.find('filename').text
        filename = filename[:-4]
        dict['filename'] = filename + '.jpg'
        
        size = root.find('size') 
        width = size.find('width').text   
        height = size.find('height').text    
        dict['image_width'] = width
        dict['image_height'] = height
        
        count = 1
        for _,object in enumerate(root.findall('object')): #找到root节点下的所有object节点 
            name = object.find('name').text   #子节点下节点name的值 
            dict['classes_text'] = ['person']
            if name == 'person':
                bndbox = object.find('bndbox')      #子节点下属性bndbox的值 
                xmin = bndbox.find('xmin').text
                ymin = bndbox.find('ymin').text
                xmax = bndbox.find('xmax').text
                ymax = bndbox.find('ymax').text
                dict['bbox' + str(count)] = [xmin, ymin, xmax, ymax]
                count += 1
                
        dict['classes'] = 1

        info_list.append(dict)
    return info_list

img_name_list = os.listdir(img_dir)
info_list = deal_xml(img_name_list)
# 将解析出来的信息保存
with open('./image_info.txt', 'w') as f:
    pickle.dump(info_list, f)
