from django.shortcuts import render

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.status import HTTP_200_OK, HTTP_404_NOT_FOUND, HTTP_400_BAD_REQUEST, HTTP_403_FORBIDDEN, \
    HTTP_202_ACCEPTED, HTTP_500_INTERNAL_SERVER_ERROR
from rest_framework.exceptions import ValidationError
from rest_framework.generics import ListCreateAPIView

from rest_framework.decorators import api_view
from django.forms.models import model_to_dict

from django.db.models import Q
from django.db import transaction
import numpy as np
from .models import Features, Scholar

from .serializers import (
    FeatureSerializer,
    ScholarSerializer
)
# Create your views here.
import requests
import json
# import tensorflow and facenet module
import tensorflow as tf
import os
import sys
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import pickle
from io import BytesIO
from .src import facenet
from .src.align import detect_face

from scipy import misc


# 训练模型的路径
MODEL_PATH = os.getcwd() + "/facemarker/src/models/20180402-114759"
# MODEL_PATH = os.getcwd() + "/facemarker/src/models/20180408-102900"
# MODEL_PATH = os.getcwd() + "/facemarker/src/models/20190218-164145"

from facemarker import pic_to_oss

class Face:
    def __init__(self):
        self.bounding_box = None
        self.image = None
        self.container_image = None
        self.embedding = None
        self.cropped_url = None

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


with tf.Graph().as_default():
    gpu_memory_fraction = 1.0
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess = tf.Session(config=config)
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

face_crop_size = 160
face_crop_margin = 32

features_data = Features.objects.all()
scholar_data = Scholar.objects.all()

with tf.Graph().as_default():
    sess = tf.Session(config=config)
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


    @api_view(['POST'])
    def feature_query(request):
        if request.method == 'POST':
            try:
                req_data = request.data
                pic_url = req_data.get('pic_url')
                img_data = get_picture_data(pic_url)
                faces = find_faces(img_data)
                print('the number of faces:', len(faces))

                query_list = []
                print("start face embedding")
                for face in faces:
                    prewhiten_face = facenet.prewhiten(face.image)
                    feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False}
                    face.embedding = sess.run(embeddings, feed_dict=feed_dict)[0]
                    # print('emb of face', face.embedding)
                    dist_array = cal_dist(features_data, face.embedding)
                    print(dist_array)
                    schloar_list=[]
                    for dist in dist_array:
                        try:
                            scholar = model_to_dict(Scholar.objects.get(id=dist['scholar_id']))
                            scholar['query_url'] = pic_url
                            scholar['dist'] = dist['dist']
                            scholar['op']=dist['op']

                            schloar_list.append(scholar)
                        except Scholar.DoesNotExist:
                            Features.objects.filter(scholar_id=dist['scholar_id']).delete()
                            continue

                    query_list.append({'scholar_list':schloar_list,
                                       'cropped_face':face.cropped_url})
                #     short_dist, user_idx, dist_array = cal_dist(features_data, face.embedding)
                #     data = {
                #         'short_dist':short_dist[0],
                #         'user_idx':user_idx[0]
                #     }
                #     query_list.append(data)
                #     print(dis_array)
                #     print('short_dist',short_dist)
                #     print('user_idx',user_idx)
                # scholar_list = []
                # for info in query_list:
                #     sid_in_features=features_data[np.asscalar(info['user_idx'])].scholar_id
                #
                #     scholar = model_to_dict(Scholar.objects.get(id=sid_in_features))
                #     scholar['query_url']=pic_url
                #     scholar['dist']=info['short_dist']
                #     scholar_list.append(scholar)

                return Response({
                    "msg":"1",
                    "result":query_list

                }, status=HTTP_200_OK)
            except ValidationError:
                raise ValidationError


    # 插入学者图片
    @api_view(['POST'])
    def feature_create(request):
        if request.method == 'POST':
            try:
                req_data = request.data
                print(req_data)
                scholar_serializer = ScholarSerializer(data = req_data)
                scholar_serializer.is_valid(raise_exception=True)
                scholar_serializer.save()
                print(scholar_serializer.data)
                # pic_url = scholar_serializer.data.get('pic_url')
                pic_url = req_data.get('pic_url')

                img_data = get_picture_data(pic_url)
                faces = find_faces(img_data) #list
                if len(faces)>1:
                    return Response({"error":"抱歉，上传的图片检测到有多张人脸"})
                elif len(faces)<1:
                    return Response({"error":"抱歉，上传的图片无法检测到人脸"})
                else:
                    face = faces[0]
                    prewhiten_face = facenet.prewhiten(face.image)
                    feed_dict = {images_placeholder: [prewhiten_face], phase_train_placeholder: False}
                    face.embedding = sess.run(embeddings, feed_dict=feed_dict)[0]
                    feature = pickle.dumps(face.embedding)
                    feature = Features.objects.create(scholar_id = scholar_serializer.data['id'],feature = feature)
                    feature = model_to_dict(feature)
                return Response({
                    'scholar': scholar_serializer.data,
                    'feature':feature
                }, status=HTTP_200_OK)
            except ValidationError:
                raise ValidationError

                return Response(scholar_serializer.data, status=HTTP_200_OK)
            except ValidationError:
                raise ValidationError


def get_picture_data(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return np.array(image)


def find_faces(image):
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    print("start find face")

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
        cropped_url = pic_to_oss.uploadImgToOss(face.image)
        face.cropped_url = cropped_url
        print(cropped_url)
        faces.append(face)

    return faces


def cal_dist(features_data, s_feature):
    scholar_id_array = []
    distance_array = []
    # print('数据库中的features', np.loads(features_data[0].feature))
    op_array=[]
    for feature_data in features_data:
        feature = feature_data.feature
        scholar_id = feature_data.scholar_id
        feature = pickle.loads(feature)
        # 计算所有检测到的人脸和查询的人脸的距离
        distance = np.sqrt(np.sum(np.square(s_feature-feature)))
        op7 = np.dot(s_feature, feature) / (np.linalg.norm(s_feature) * (np.linalg.norm(feature)))
        #
        #
        # dist =  {
        #     'dist':distance,
        #     'scholar_id':scholar_id,
        # }
        op = {
            'op': op7,
            'scholar_id': scholar_id,
            'dist': distance,
        }
        # distance_array.append(dist)
        op_array.append(op)
    op_array.sort(key=lambda k: (k.get('op', 0)),reverse=True)
    # distance_array.sort(key=lambda k: (k.get('dist', 0)))


    # distance_array = np.vstack(distance_array)
    # distance_array.sort()
    # short_dist = np.min(distance_array, axis=1)
    # user_idx = np.argmin(distance_array, axis=1)

    # return short_dist, user_idx,distance_array
    # return distance_array[:3]
    return op_array[:3]