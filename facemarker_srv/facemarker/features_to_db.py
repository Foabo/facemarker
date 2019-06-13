import requests
import urllib.request
from bs4 import BeautifulSoup
import numpy as np
import time
from urllib.parse import urlparse
import re
import os, sys
import numpy as np

# Create your views here.
import requests
import json
# import tensorflow and facenet module
import tensorflow as tf
import os
import sys
import cv2
from PIL import Image
import pickle
from io import BytesIO

# from rest_framework.exceptions import ValidationError
from django.forms import model_to_dict
from scipy import misc
import uuid

# 训练模型的路径


os.chdir("..")
a = os.getcwd()



MODEL_PATH = os.getcwd() + "/facemarker/src/models/20180402-114759"
# MODEL_PATH = os.getcwd() + "/facemarker/src/models/20180408-102900"
# MODEL_PATH = os.getcwd() + "/facemarker/src/models/20190218-164145"
# MODEL_PATH = os.getcwd() +"/facemarker/src/models/20190515" 模型无法使用





sys.path.append(a)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'facemarker_srv.facemarker_srv.settings')

import django

django.setup()

pic_path = os.walk(a + '/pictures/')


class Face:
    def __init__(self):
        self.bounding_box = None
        self.image = None
        self.container_image = None
        self.embedding = None


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

from facemarker.models import Features, Scholar
import pickle
from facemarker.src import facenet
from facemarker.src.align import detect_face
from facemarker.src import config
from facemarker.src import facetask

# with tf.Graph().as_default():
#     gpu_memory_fraction = 1.0
#     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
#     sess = tf.Session(config=config)
#     with sess.as_default():
#         pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
with tf.Graph().as_default():
    gpu_memory_fraction = 1.0
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

face_crop_size = 160
face_crop_margin = 32

with tf.Graph().as_default():
    sess = tf.Session()
    # src.facenet.load_model(modelpath)
    # 加载模型
    meta_file, ckpt_file = facenet.get_model_filenames(MODEL_PATH)
    saver = tf.train.import_meta_graph(os.path.join(MODEL_PATH, meta_file))
    saver.restore(sess, os.path.join(MODEL_PATH, ckpt_file))
    # Get input and output tensors
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

    # 进行人脸识别，加载
    print('Creating networks and loading parameters')


    def main():
        for filename in os.listdir(a + '/pictures/'):
            if not filename.startswith('.'):
                organization = filename

            for scholar_name in os.listdir(a + '/pictures/' + organization + '/'):
                if not scholar_name.startswith('.'):
                    # print(scholar_name)
                    pic_path = a + '/pictures/' + organization + '/' + scholar_name + '/'
                    pic_list = os.listdir(pic_path)

                    if (len(pic_list)):
                        pic = pic_list[0]
                        for pic in pic_list:
                            if not pic.startswith('.'):
                                try:
                                    scholar_data = Scholar.objects.get(organization=organization, name=scholar_name)
                                except Scholar.DoesNotExist:
                                    scholar_data = Scholar.objects.create(organization=organization, name=scholar_name)
                                print(model_to_dict(scholar_data))
                                pic_name = pic_path + pic
                                print(pic_name)
                                img_data = misc.imread(pic_name, mode='RGB')
                                try:
                                    faces = find_faces(img_data)

                                except IndexError:
                                    continue
                                if len(faces):
                                    face = faces[0]

                                    prewhiten_face = facenet.prewhiten(face.image)
                                    feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False}
                                    face.embedding = sess.run(embeddings, feed_dict=feed_dict)[0]

                                    feature = pickle.dumps(face.embedding)
                                    # feature = json.dumps(feature)
                                    feature, created = Features.objects.get_or_create(id=uuid.uuid4(),
                                                                                      scholar_id=scholar_data.id,
                                                                                      defaults={"feature": feature})




#

def get_picture_data(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return np.array(image)


def find_faces(image):
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.9]  # three steps's threshold
    factor = 0.709  # scale factor

    print('-----------------find face-----------------')

    faces = []
    bounding_boxes, _ = detect_face.detect_face(image, minsize,
                                                pnet, rnet, onet,
                                                threshold, factor)
    for bb in bounding_boxes:
        face = Face()
        face.container_image = image
        face.bounding_box = np.zeros(4, dtype=np.int32)

        img_size = np.asarray(image.shape)[0:2]
        face.bounding_box[0] = np.maximum(bb[0] - face_crop_margin / 2, 0)
        face.bounding_box[1] = np.maximum(bb[1] - face_crop_margin / 2, 0)
        face.bounding_box[2] = np.minimum(bb[2] + face_crop_margin / 2, img_size[1])
        face.bounding_box[3] = np.minimum(bb[3] + face_crop_margin / 2, img_size[0])
        cropped = image[face.bounding_box[1]:face.bounding_box[3], face.bounding_box[0]:face.bounding_box[2], :]
        face.image = misc.imresize(cropped, (face_crop_size, face_crop_size), interp='bilinear')

        faces.append(face)

    return faces


def debug():
    pic_name = '/Users/inf/project/face/facemarker_srv/pictures/北京科技大学/金田/4117adf02af5268dac3eabcc71ace66f.jpg'
    img_data = misc.imread(pic_name, mode='RGB')
    faces = []
    try:
        faces = find_faces(img_data)

    except IndexError:
        return IndexError
    for face in faces:
        print(face.embedding)


if __name__ == '__main__':
    main()
    # debug()
