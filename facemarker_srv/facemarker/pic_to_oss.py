import requests
import json
import oss2
import os
import uuid
from PIL import Image
from io import BytesIO


os.chdir("..")
a = os.getcwd()

# AliCloud access key ID
OSS_ACCESS_KEY_ID = "LTAIuOqaCY6lKUPh"

# AliCloud access key secret
OSS_ACCESS_KEY_SECRET = "SxgBk6hhe7XJqiASiRnnrXUUPEcetO"

# The name of the bucket to store files in
OSS_BUCKET_NAME = "pic-so-link"

# The URL of AliCloud OSS endpoint
OSS_END_POINT = "oss-cn-shenzhen.aliyuncs.com"

# The host of AliCloud OSS
OSS_HOST = 'https://cdn.so-link.org/'

# The callback url
OSS_CALLBACK_URL = 'https://debug.so-link.org/oss_storage/callback'

access_key_id = os.getenv('OSS_TEST_ACCESS_KEY_ID', OSS_ACCESS_KEY_ID)
access_key_secret = os.getenv('OSS_TEST_ACCESS_KEY_SECRET', OSS_ACCESS_KEY_SECRET)
bucket_name = os.getenv('OSS_TEST_BUCKET', OSS_BUCKET_NAME)
endpoint = os.getenv('OSS_TEST_ENDPOINT', OSS_END_POINT)

auth = oss2.Auth(access_key_id, access_key_secret)
# Endpoint以杭州为例，其它Region请按实际情况填写。
bucket = oss2.Bucket(auth, endpoint, bucket_name)



"""
上传图片到阿里云oss服务器
"""
def uploadFileToOss(img_path):
    """

    :param img_path:本地的学者图片路径
    :return: 回调函数返回的数据
    """


    # put_object/complete_multipart_upload支持上传回调，resumable_upload不支持。
    # 回调服务器(callbacke server)的示例代码请参考 http://shinenuaa.oss-cn-hangzhou.aliyuncs.com/images/callback_app_server.py.zip
    # 您也可以使用OSS提供的回调服务器 http://oss-demo.aliyuncs.com:23450，调试您的程序。调试完成后换成您的回调服务器。
    # 首先初始化AccessKeyId、AccessKeySecret、Endpoint等信息。
    # 通过环境变量获取，或者把诸如“<你的AccessKeyId>”替换成真实的AccessKeyId等。

    try:


        #上传的文件名称
        key='scholars/'+str(uuid.uuid4())+'.png'
        #拼接得到二维码图片地址
        img_url=OSS_HOST+key

        """
        put_object上传回调
        """

        # 准备回调参数，更详细的信息请参考 https://help.aliyun.com/document_detail/31989.html
        callback_dict = {}
        callback_dict['callbackUrl'] = 'https://api.so-link.org/oss_storage/callback'
        callback_dict[
            'callbackBody'] = 'bucket=${bucket}&object=${object}&size=${size}&mimeType=${mimeType}&height=${imageInfo.height}&width=${imageInfo.width}'
        callback_dict['callbackBodyType'] = 'application/x-www-form-urlencoded'
        # 回调参数是json格式，并且base64编码
        callback_param = json.dumps(callback_dict).strip()
        base64_callback_body = oss2.utils.b64encode_as_string(callback_param)
        # 回调参数编码后放在header中传给oss
        headers = {'x-oss-callback': base64_callback_body}

        # 上传并回调
        # result = bucket.put_object(key, content, headers)
        result= bucket.put_object_from_file(key,img_path,headers)
        s=result.resp.read()
        print(s)
        return img_url
    except Exception as e:
        print(e)

def uploadImgToOss(img_data):
    try:
        # 上传的二进制的内容
        key='croped-face/'+str(uuid.uuid4())+'.jpeg'
        cropped_url=OSS_HOST+key
        tmp = Image.fromarray(img_data)
        fd = BytesIO()
        tmp.save(fd, format='jpeg')
        fd.seek(0)
        bucket.put_object(key, fd)
        return cropped_url
    except Exception as e:
        print(e)

if __name__ == '__main__':
    r =  uploadFileToOss('/Users/inf/project/base_facenet/1.jpg')
    print(r)