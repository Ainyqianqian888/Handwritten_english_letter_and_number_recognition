import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

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