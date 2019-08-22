---
title: tensorflow-serving部署
data: 2019-8-22
categories: web
---

#### tensorflow-seving简介

TensorFlow Serving 是一个用于机器学习模型 serving 的高性能开源库。它可以将训练好的机器学习模型部署到线上，可以使用 gRPC或rest api作为接口接受外部调用。它支持模型热更新与自动模型版本管理。这意味着一旦部署 TensorFlow Serving 后，你再也不需要为线上服务操心，只需要关心你的线下模型训练。



#### docker

Docker是一个虚拟环境容器，可以将你的开发环境、代码、配置文件等一并打包到这个容器中，并发布和应用到任意平台中。

###### 安装

安装地址[https://docs.docker.com/install/linux/docker-ce/ubuntu](https://docs.docker.com/install/linux/docker-ce/ubuntu)

在docker中想要使用gpu的话，需要安装nvidia-docker。安装地址[https://github.com/NVIDIA/nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

为了使用gpu，还需要安装nvidia-container-runtime，安装地址[https://github.com/NVIDIA/nvidia-container-runtime](https://github.com/NVIDIA/nvidia-container-runtime)

###### 拉取镜像文件

```python
docker pull tensorflow/serving:latest-gpu
```

上面命令为拉取最新的tensorflow-serving镜像文件，但是最新的镜像需要cuda10的支持，为了使用cuda9对应的镜像文件，可以指定拉取的tensorflow-serving的版本。

```python
docker pull tensorflow/serving:1.12.0-gpu
```

###### docker的简单操作

```python
#查看当前所有的容器状态
sudo docker ps -a
#停止某个运行的容器
sudo docker rm '容器id'
#重启某个运行的容器
sudo docker start/restart '容器id'
#删除某个容器
sudo docker rm '容器id'
```



#### 部署inceptionv3分类应用到tensorflow-serving的实例

###### 下载keras中具有imagenet权重的inceptionv3模型

```python
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input

inception_model = InceptionV3(weights='imagenet', input_tensor=Input(shape=(224, 224, 3)))
inception_model.save('inception.h5')
```

###### 将keras格式的模型文件以tensorflow-serving可以处理的格式导出

```python
import tensorflow as tf

# The export path contains the name and the version of the model
tf.keras.backend.set_learning_phase(0)  # Ignore dropout at inference
model = tf.keras.models.load_model('./inception.h5')
export_path = '../my_image_classifier/1'

# Fetch the Keras session and save the model
# The signature definition is defined by the input and output tensors
# And stored with the default serving key
with tf.keras.backend.get_session() as sess:
    tf.saved_model.simple_save(
        sess,
        export_path,
        inputs={'input_image': model.input},
        outputs={t.name: t for t in model.outputs})
```

###### 启动tensorflow-serving服务

```python
#source必须为绝对路径
sudo docker run --runtime=nvidia -p 8501:8501 --mount type=bind,source=/home/kepa/hsy_workspace/web/web-flask/my_image_classifier,target=/models/my_image_classifier -e MODEL_NAME=my_image_classifier -t tensorflow/serving:1.12.0-gpu & 
```

###### 创建flask服务器

创建flask服务器的原因在于：

- 执行图像预处理的部分。
- 如果我们打算提供多个模型，那么我们不得不创建多个 TensorFlow Serving 服务并且在前端代码添加新的 URL。但 Flask 服务会保持域 URL 相同，而我们只需要添加一个新的路由（一个函数）。

创建flask服务器代码app.py如下：

```python
import base64
import json
from io import BytesIO

import numpy as np
import requests
from flask import Flask, request, jsonify
from keras.applications import inception_v3
from keras.preprocessing import image

# from flask_cors import CORS

app = Flask(__name__)


# Uncomment this line if you are making a Cross domain request
# CORS(app)

# Testing URL
@app.route('/hello/', methods=['GET', 'POST'])
def hello_world():
    return 'Hello, World!'


@app.route('/imageclassifier/predict/', methods=['POST'])
def image_classifier():
    # Decoding and pre-processing base64 image
    img = image.img_to_array(image.load_img(BytesIO(base64.b64decode(request.form['b64'])),
                                            target_size=(224, 224))) / 255.

    # this line is added because of a bug in tf_serving(1.10.0-dev)
    img = img.astype('float16')

    # Creating payload for TensorFlow serving request
    payload = {
        "instances": [{'input_image': img.tolist()}]
    }

    # Making POST request
    r = requests.post('http://localhost:9000/v1/models/ImageClassifier:predict', json=payload)

    # Decoding results from TensorFlow Serving server
    pred = json.loads(r.content.decode('utf-8'))

    # Returning JSON response to the frontend
    return jsonify(inception_v3.decode_predictions(np.array(pred['predictions']))[0])
```

在启动flask服务之前需要指定FLASK_APP环境变量，然后运行启动程序。

```python
export FLASK_APP=app.py
flask run --host=0.0.0.0  #ip地址指定为0.0.0.0允许从其他机器访问服务器
```

###### 测试请求

```python
# importing the requests library
import argparse
import base64

import requests

# defining the api-endpoint
API_ENDPOINT = "http://localhost:5000/imageclassifier/predict/"

# taking input image via command line
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path of the image")
args = vars(ap.parse_args())

image_path = args['image']
b64_image = ""
# Encoding the JPG,PNG,etc. image to base64 format
with open(image_path, "rb") as imageFile:
    b64_image = base64.b64encode(imageFile.read())

# data to be sent to api
data = {'b64': b64_image}

# sending post request and saving response as response object
r = requests.post(url=API_ENDPOINT, data=data)

# extracting the response
print("{}".format(r.text))
```



