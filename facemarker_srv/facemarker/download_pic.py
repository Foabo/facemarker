import requests
import urllib.request
from bs4 import BeautifulSoup

import time
from urllib.parse import urlparse
import re
import os, sys

os.chdir("..")
a = os.getcwd()

sys.path.append(a)
print(sys.path)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'facemarker_srv.facemarker_srv.settings')

import django

django.setup()

from facemarker.models import Features, Scholar


def download(scholar_data):
    # url = 'http://scit.bjtu.edu.cn/cms/staff/7922/?cat=12'
    url = scholar_data.homepage
    print(url)
    domain = urlparse(url)
    scheme = domain.scheme
    netloc = domain.netloc
    print(domain)
    headers = {
        "Accept-Encoding": "gzip, deflate",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",

        "Content-Type": "text/plain",
        "Cookie": "SESSIONID=89A35CD190340BE6849FC3B5494D827D",
        "Pragma": "no-cache",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.103 Safari/537.36",

    }

    try:
        # 一下except都是用来捕获当requests请求出现异常时，
        # 通过捕获然后等待网络情况的变化，以此来保护程序的不间断运行
        response = requests.get(url, headers=headers, timeout=3, )  # 使用headers避免访问受限
        # response = urllib.request.urlopen(url).read().decode('utf-8')


    except requests.exceptions.ConnectionError:
        print('ConnectionError -- please wait 3 seconds')
        time.sleep(2)
        return

    except requests.exceptions.ChunkedEncodingError:
        print('ChunkedEncodingError -- please wait 3 seconds')
        time.sleep(2)
        return

    except:
        print('Unfortunitely -- An Unknow Error Happened, Please wait 3 seconds')
        time.sleep(2)
        return

    if len(response.text) < 1:
        return

    print('接受返回')
    response.encoding = 'utf-8'
    soup = BeautifulSoup(response.content, 'lxml')
    items = soup.find_all('img')

    name = scholar_data.name
    organization = scholar_data.organization
    folder_path = os.getcwd() + "/data/" + organization + '/' + name + '/'
    print(folder_path)
    print('img 个数', len(items))
    if os.path.exists(folder_path) == False:  # 判断文件夹是否已经存在
        os.makedirs(folder_path)  # 创建文件夹

    # with open(folder_path + organization + '.txt', "a") as file:
    #     file.flush()
    #     file.close()
    # print('file close')
    for index, item in enumerate(items):
        if item:
            print(item)
            src = item.get('src')
            print(src)
            pattern = None
            if src is not None:
                pattern = re.match(r'(.*?(\.jpg|\.png|\.jpeg)).*', src)

            if pattern:
                if src.startswith('http'):
                    print('startswith http')
                    html = requests.get(src)  # get函数获取图片链接地址，requests发送访问请求

                elif src.startswith('.'):
                    continue

                else:
                    src = scheme + '://' + netloc + src
                    print('=============src', src)
                    html = requests.get(src)  # get函数获取图片链接地址，requests发送访问请求

                print(html)
                img_name = folder_path + str(index + 1) + '.png'

                with open(img_name, 'wb') as file:  # 以byte形式将图片数据写入
                    file.write(html.content)
                    file.flush()
                file.close()  # 关闭文件
                print('第%d张图片下载完成' % (index + 1))
                time.sleep(1)  # 自定义延时


def main():
    scholars_data = Scholar.objects.all()
    # print(scholars_data)
    # download(scholars_data[6])
    for scholar_data in scholars_data:
        download(scholar_data)


def test():
    # pattern = re.compile(r'(.*?(\.jpg|\.png|\.jpeg)).*')
    # url = 'http://sdcs.sysu.edu.cn/sites/sdcs.live1.dpcms8.sysu.edu.cn/files/styles/teacher_pic/public/zhfg.jpg?itok=b1p-zxvR'
    # pattern = re.match(r'(.*?(\.jpg|\.png|\.jpeg)).*', None)
    #
    # # m = pattern.match(url)
    # print(pattern)
    import numpy as np
    str = Features.objects.all()[0]
    str = str.feature
    str = str.spl
    print(str)

if __name__ == '__main__':
    # main()

    test()
