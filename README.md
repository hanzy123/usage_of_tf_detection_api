## 克隆仓库
`git clone https://github.com/tensorflow/models.git`

## 数据集制作脚本使用注意点

### extract_info_from_voc.py

- 该脚本用于提取voc格式的数据集的annotation信息
- 使用时要注意如果数据集的文件夹结构不是完全按照voc的来，需要修改内部部分代码：因为voc是通过图片名去对应各自的annotation文件的
- 执行完毕后会将提取出来的annotation信息dump到当前文件夹下。

### make_tf_record.py

- 该脚本是将上一步dump出来的信息进一步制作成训练需要的tfrecord文件所用
- 该脚本执行完毕会在当前目录生成`train.tfrecord`文件

## 训练

`cd models/research`

训练命令`python object_detection/train.py --pipeline_config_path=object_detection/Human_detection/train_config.txt --train_dir=object_detection/Human_detection/output/`

- `pipeline_config_path`指的是训练使用模型所对应的配置文件，在该文件中需要配置训练时候用到的超参数以及我们上一步准备的record文件路径、label_map路径、以及我们需要微调模型的路径。
- `train_dir`指的是训练后保存路径

更详细的训练操作以及文件的链接`https://note.youdao.com/share/?id=21e28781e652e0d546cd686814560ba0&type=note#/`
