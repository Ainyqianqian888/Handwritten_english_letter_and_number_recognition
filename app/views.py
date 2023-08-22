#!/usr/bin/python3
# coding:utf-8
from flask import render_template,json,jsonify,request
import cv2
from app import app
import base64
import tensorflow as tf
import numpy as np
# import tensorflow.contrib.slim as slim
# import pickle
import os
from PIL import Image,ImageFont, ImageDraw
from alfred.utils.log import logger as logging
from app.model import ResNet_inference

__global_times = 0
__checkpoint_dir = './app/train_model/checkpoint/'          # 模型文件路径


target_size = 28
num_classes = 62
# use_keras_fit = False
use_keras_fit = True
ckpt_path = r'app/checkpoints/ResNet/ResEpoch-{epoch}.ckpt'

def load_characters():
    a = open(r'app/characters.txt', 'r', encoding='UTF-8').readlines()
    return [i.strip() for i in a]
characters = load_characters() # 载入标签向量矩阵

__test_image_file  = 'app/image/pred1.png'  # 测试图片
__pred1_image_file = 'app/image/pred1.png'  # 预测结果1图片
# __pred2_image_file = './app/image/pred2.png'  # 预测结果2图片
# __pred3_image_file = './app/image/pred3.png'  # 预测结果3图片

def predictPrepare():
    # init model
    model = ResNet_inference((28, 28, 1), num_classes, 0.5)
    logging.info('model loaded.')

    latest_ckpt = tf.train.latest_checkpoint(os.path.dirname(ckpt_path))
    if latest_ckpt:
        start_epoch = int(latest_ckpt.split('-')[1].split('.')[0])
        model.load_weights(latest_ckpt)
        logging.info('model resumed from: {} at epoch: {}'.format(latest_ckpt, start_epoch))
        return model
    else:
        logging.error('can not found any checkpoints matched: {}'.format(ckpt_path))

#--------------------------#
# 对读取的图片预处理
#--------------------------#
def pre_pic(picName):
    # reIm = picName.resize((target_size,target_size), Image.ANTIALIAS)
    im_arr = np.array(picName)
    # 对图片做二值化处理（滤掉噪声，threshold调节阈值）
    threshold = 25
    # 模型的要求是黑底白字，但输入的图是白底黑字，所以需要对每个像素点的值改为255减去原值以得到互补的反色。
    for i in range(28):
        for j in range(28):
            im_arr[i][j] = 255 - im_arr[i][j]
            if (im_arr[i][j] < threshold):
                im_arr[i][j] = 0
            else: im_arr[i][j] = 255
    # 把图片形状拉成1行784列，并把值变为浮点型（因为要求像素点是0-1 之间的浮点数）
    nm_arr = im_arr.reshape([1, 784])
    nm_arr = nm_arr.astype(np.float32)
    # 接着让现有的RGB图从0-255之间的数变为0-1之间的浮点数
    img_ready = np.multiply(nm_arr, 1.0/255.0)
    img_ready = tf.reshape(img_ready,[28, 28])
    # plt.imshow(img_ready)
    # plt.show()
    return img_ready

def predict(model, img_f):
    img_ready = cv2.imread(img_f)
    img = tf.expand_dims(img_ready[:, :, 0], axis=-1)
    img = tf.image.resize(img, (target_size, target_size))

    img = pre_pic(img)

    img = tf.expand_dims(img, axis=-1)
    img = tf.transpose(img)  # 训练集集预处理没做好，这里需要旋转镜像
    # img = tf.image.rot90(img, k=1)
    img = tf.expand_dims(img, axis=-1)
    print(img.shape)
    result = model.predict(img)
    print('predict: {}'.format(characters[np.argmax(result[0])]))

    name = 'app/assets/pred_{}.png'.format(characters[np.argmax(result[0])])
    # cv2.imwrite(name, ori_img)  #路径中文会乱码
    cv2.imencode('.jpg', img_ready)[1].tofile(name)  # 正确的解决办法
    return characters[np.argmax(result[0])]


def createImage(predword,imagepath):
    im = Image.new("RGB", (64, 64), (255, 255, 255))
    dr = ImageDraw.Draw(im)
    fonts = ImageFont.truetype("app/static/fonts/msyh.ttc",36,encoding='utf-8')
    dr.text((15, 10), predword,font=fonts,fill="#000000")
    im.save(imagepath)

@app.route('/')
@app.route('/index')
def index():
    global __global_times
    global model
    if (__global_times == 0):
        model = predictPrepare()  # 加载模型，准备好预测
        __global_times = 1
    else:
        pass
    return render_template("index.html",title='Home')

@app.route('/chineseRecognize',methods=['POST'])
def chineseRecognize():
    # 接受前端发来的数据
    data = json.loads(request.form.get('data'))
    imagedata = data["test_image"]
    imagedata = imagedata[22:]
    img = base64.b64decode(imagedata)
    file = open(__test_image_file, 'wb')
    file.write(img)
    file.close()

    predict_word = predict(model, __test_image_file)
    createImage(predict_word, __pred1_image_file) # 生成准确率top1的图片
        # createImage(word_dict[predict_index[0][1]], __pred2_image_file)
        # createImage(word_dict[predict_index[0][2]], __pred3_image_file)
    # else:
        # a = cv2.imread(__test_image_file)
        # cv2.imshow('monitor', a)
        # cv2.moveWindow("monitor", 960, 540)
        # predict_word = predict(model, __test_image_file)
        # createImage(predict_word, __pred1_image_file)
        # createImage(word_dict[predict_index[0][1]], __pred2_image_file)
        # createImage(word_dict[predict_index[0][2]], __pred3_image_file)

    # 将识别图片转码传给前端，并带上对应的准确率
    with open(__pred1_image_file, 'rb') as fin:
        image1_data = fin.read()
        pred1_image = base64.b64encode(image1_data)
    # with open(__pred2_image_file, 'rb') as fin:
    #     image2_data = fin.read()
    #     pred2_image = base64.b64encode(image2_data)
    # with open(__pred3_image_file, 'rb') as fin:
    #     image3_data = fin.read()
    #     pred3_image = base64.b64encode(image3_data)
    info = dict()
    info['pred1_image'] = "data:image/jpg;base64," + pred1_image.decode()
    info['pred1_accuracy'] = str('{:.2%}'.format(0.9))
    # info['pred2_image'] = "data:image/jpg;base64," + pred2_image.decode()
    # info['pred2_accuracy'] = str('{:.2%}'.format(predict_val[0][1]))
    # info['pred3_image'] = "data:image/jpg;base64," + pred3_image.decode()
    # info['pred3_accuracy'] = str('{:.2%}'.format(predict_val[0][2]))
    return jsonify(info)