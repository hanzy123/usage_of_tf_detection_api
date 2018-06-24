这个任务是用我们自己的数据集，利用tensorflow提供的检测api来完成行人检测任务。

## 数据集准备

我们使用VOC数据集中的行人图片，这里介绍一下:
- 如何从voc数据集中提取我们需要的行人类别相关信息，以及提取之后的保存格式
- 如何将提取出来的信息转化成tensorflow训练所需要的TF-record类型。

### 从voc2007数据集中提取person类别的相关信息

提取后信息的保存格式：
`{key:filename, values:image_width, image_height, bboxN:[xmins, ymins,  xmaxs, ymaxs], classes_text, classes}`

`example:{filename:000001.jpg, width:640, height:320, bbox1:[20, 30, 100, 100],classes_text:['Person'], classes:[1] }`

**voc数据集提取person信息代码**
```
## 提取person类人的相关信息
info_list = []
def deal_xml(image_list):
    for image_name in image_list:
        dict = {}
        xml_path = os.path.join(root_path,'train/Annotations',image_name + '.xml')
        # train 和 test 分开制作
        #xml_path = os.path.join(root_path,'test/Annotations',image_name + '.xml')
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
```

**生成TFRecord代码**

```
def create_tf_example(example):
    # example是上面代码得到的info_list中的每个元素
    # TODO(user): Populate the following variables from your example.
    height = int(example['image_height']) # Image height
    width = int(example['image_width']) # Image width
    filename = example['filename'] # Filename of the image. Empty if image is not from file
    #encoded_image_data = None # Encoded image bytes
    # 原始图片格式
    image_format = 'jpg' # b'jpeg' or b'png'
    
    image_filepath = '/root/dl-data/Human_detection_datasets/VOC2007/train/JPEGImages/' + filename
    #image_filepath = '/root/dl-data/Human_detection_datasets/VOC2007/test/JPEGImages/' + filename
    with tf.gfile.Open(image_filepath, "r") as f:
        encoded_image_data = f.read()
    
    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box
             # (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
    classes_text = []
    classes = []
    
    bbox_id = 1
    while True:
        try:
            bbox = example['bbox' + str(bbox_id)]
            xmins.append(int(bbox[0])*1.0 / int(width))
            xmaxs.append(int(bbox[2])*1.0 / int(width))
            ymins.append(int(bbox[1])*1.0 / int(height))
            ymaxs.append(int(bbox[3])*1.0 / int(height))
            classes_text.append('person') # List of string class name of bounding box (1 per box)
            classes.append(1) # List of integer class id of bounding box (1 per box)
            bbox_id += 1
        except:
            break
    
    tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter('./train.tfrecord')
    #writer = tf.python_io.TFRecordWriter('./eval.tfrecord')
    # TODO(user): Write code to read in your dataset to examples variable

    # 对上个模块的代码得到的info_list进行遍历处理
    for example in human_image_list_trainval:
    #for example in human_image_list_test:
    
        tf_example = create_tf_example(example)
        writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    tf.app.run()
```
到这里我们已经完成了训练数据集的准备。

## 模型训练

### 训练时候需要准备的文件

- label_map.pbtxt
```
item {
  id: 1
  name: 'people'
}
item {
  id: 2
  name: 'dog'
}
...
```
++这里要注意一下id要从1开始++

- train_config.txt

`train_config.txt`文件中记录了训练需要的相关信息，包括模型路径，数据集路径和`label_map.txt`路径

这里给出不同的模型对应的`train_config.txt`  [train_configs](https://github.com/tensorflow/models/tree/fd7b6887fb294e90356d5664724083d1f61671ef/research/object_detection/samples/configs)

以及各种现在tensorflow提供的检测模型链接 [model_zoo](https://github.com/tensorflow/models/blob/fd7b6887fb294e90356d5664724083d1f61671ef/research/object_detection/g3doc/detection_model_zoo.md)

- output文件夹

另外需要手动生成一个output文件夹来存放训练产生的checkpoint

>python object_detection/train.py --pipeline_config_path=object_detection/Human_detection/train_config.txt --train_dir=object_detection/Human_detection/output/

**训练时候碰到的问题**

- 缺少文件from object_detection.protos import preprocessor_pb2

github上提问 [I can't find preprocessor_pb2,who can help me](https://github.com/tensorflow/models/issues/2084)

解决方案:

原因:没有编译Protobuf Compilation

```
# From tensorflow/models/
protoc object_detection/protos/*.proto --python_out=.
```
然后
```
# From tensorflow/models/
# $PYTHONPATH = /root/anaconda3/lib/python3.6/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

最后使用时候需要
`os.path.insert(0, '/root/dl-data/github/models')`
最后训练得到三个`.data-00000-of-00001、.index、.meta`文件
