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

sys.path.append(a)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'facemarker_srv.facemarker_srv.settings')

import django

django.setup()

pic_path = os.walk(a + '/pictures/')
from facemarker.models import Features, Scholar
import pickle
from facemarker import pic_to_oss

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
                    try:
                        scholar_data = Scholar.objects.get(organization=organization, name=scholar_name)
                    except Scholar.DoesNotExist:
                        scholar_data = Scholar.objects.create(organization=organization, name=scholar_name)
                    print(model_to_dict(scholar_data))
                    pic_name = pic_path + pic
                    print(pic_name)
                    # img_data = misc.imread(pic_name, mode='RGB')
                    pic_url = pic_to_oss.uploadFileToOss(pic_name)
                    print(pic_url)
                    scholar_data.pic_url = pic_url
                    scholar_data.save()

if __name__ == '__main__':
    main()