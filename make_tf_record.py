import pickle

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
    
    # 载入上一步保存出来的pickle信息
    image_info = []
    with open('./image_info.txt', 'r') as f:
        image_info = pickle.load(f)
    
    human_image_list_trainval = image_info
    
    # 对上个模块的代码得到的image_info进行遍历处理
    for example in human_image_list_trainval:
    #for example in human_image_list_test:
    
        tf_example = create_tf_example(example)
        writer.write(tf_example.SerializeToString())

    writer.close()
    print("tfrecord gen ok!")

if __name__ == '__main__':
    tf.app.run()
