# TensorFlow 2 EMNIST数据集上的ResNet手写字母数字识别模型
本项目是在对Resnet在识别手写英文字母和数字场景下的的部署。

## 数据集

数据全部来自于TF官网收录的EMNIST数据集：
- https://tensorflow.google.cn/datasets/catalog/emnist?hl=en
- emnist/byclass (default config)：本项目使用默认的62分类数据集，分别为10个数字，26个大小写字母（10+26+26），训练集697932个，测试集116323个

## ResNet

采用ResNet模型进行训练，模型构建写在model.py中。


```python
#--------------------------#
# RESNET
#--------------------------#
def res_net_block(input_data, filters, conv_size):
  # CNN层
  x = layers.Conv2D(filters, conv_size, activation='relu', padding='same')(input_data)
  x = layers.BatchNormalization()(x)
  x = layers.Conv2D(filters, conv_size, activation=None, padding='same')(x)
  # 第二层没有激活函数
  x = layers.BatchNormalization()(x)
  # 两个张量相加
  x = layers.Add()([x, input_data])
  # 对相加的结果使用ReLU激活
  x = layers.Activation('relu')(x)
  # 返回结果
  return x

def ResNet_inference(input_shape, n_classes, dropout):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation='relu')(inputs)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.MaxPooling2D(3)(x)
    num_res_net_blocks = 10
    for i in range(num_res_net_blocks):
        x = res_net_block(x, 64, 3)
    # 添加一个CNN层
    x = layers.Conv2D(64, 3, activation='relu')(x)
    # 全局平均池化GAP层
    x = layers.GlobalAveragePooling2D()(x)
    # 几个密集分类层
    x = layers.Dense(256, activation='relu')(x)
    # 退出层
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    res_net_model = keras.Model(inputs, outputs)
    res_net_model.compile(optimizer=keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    res_net_model.summary()
    #  105s 53ms/step - loss: 0.2584 - accuracy: 0.8978 - val_loss: 0.3838 - val_accuracy: 0.8743
    #  2000*20steps开始过拟合
    return res_net_model
```

## 加载模型
项目采用https://github.com/Ainyqianqian888/Handwritten_english_letter_and_number_recognition/tree/master 所训练的模型，详细原理可参考该链接，训练好的模型保存在checkpoint文件夹中。


## 环境
- python               3.7.7
- alfred               0.3                
- alfred-py            3.0.7                          
- Flask                2.2.5
- numpy                1.21.6     
- opencv-python        4.8.0.76                      
- rich                 13.5.2            
- tensorflow           2.1.0

## 模型部署
使用`python+flask`搭建的一个网站，然后从网页的写字板上获取鼠标手写的字母或数字经过转码后传回后台，并经过图片裁剪处理之后传入`ResNet`模型中进行识别，最后通过`PIL`将识别结果生成图片，最后异步回传给web端进行识别结果展示。 
这里对英文字母和数字总共`36`个字进行识别。   
<br>老师，不好意思，目前在本地搭建的
![demogif](https://github.com/Ainyqianqian888/Handwritten_english_letter_and_number_recognition/blob/master/demo.gif) <br>

## 运行
 1、下载项目代码，安装项目所需的库；<br>
 2、运行`python run.py`；<br>
 4、打开本地浏览器输入`localhost:5000`进行查看；<br>
 
