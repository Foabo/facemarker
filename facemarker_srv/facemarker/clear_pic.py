import os
import sys
from PIL import Image
from libtiff import TIFF
from scipy import misc
# from wand.image import Image
import cv2
os.chdir("..")
a = os.getcwd()
# /Users/inf/project/face/facemarker_srv
import  re
def tiff_to_image_array(tiff_image_name, out_folder, out_type):
    ##tiff文件解析成图像序列
    ##tiff_image_name: tiff文件名；
    ##out_folder：保存图像序列的文件夹
    ##out_type：保存图像的类型，如.jpg、.png、.bmp等
    tif = TIFF.open(tiff_image_name, mode="r")
    idx = 0
    for im in list(tif.iter_images()):
        #
        im_name = out_folder + str(idx) + out_type
        misc.imsave(im_name, im)
        print(im_name)
        idx = idx + 1
    return


def main():
    path = a + '/Users/inf/project/face/facemarker_srv/scholars'
    pose_png = '/Users/inf/project/face/facemarker_srv/scholars/scholar_160_jpg'
    g = os.walk(path)
    # os.path.join(path, file_name)
    for path, dir_list, file_list in g:
        # print(path)
        g = os.walk(path)
        for _path, _dir_list, _file_list in g:
            for dir in _dir_list:
                print(path)
                print(dir)

                folder = os.path.join(pose_png, dir)
                if not os.path.exists(folder):  # 判断当前路径是否存在，没有则创建new文件夹
                    os.makedirs(folder)
                i= 1
                for filename in os.listdir(path + '/' + dir):
                    img_name = os.path.join(path, dir, filename)
                    name_split = filename.split('.')
                    tif_name = os.path.join(folder, dir+'_%04d'%i + '.jpg')
                    print(tif_name)
                    img = cv2.imread(img_name)
                    cv2.imwrite(tif_name, img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
                    i=i+1

                    # for dir in dir_list:
#     print(dir)
# for filename in os.listdir(a + '/'+dir+'/'):
#     print(filename)

# for file_name in file_list:
#     name_split = file_name.split('.')
#     print(os.path.join(path, file_name))
# with Image(filename=os.path.join(path, file_name)) as img:
#     print(1)
#     # img.format('jpg')
#     img.save(filename=os.path.join(pose_png, name_split[0] + '.png'))  # png, jpg, bmp, gif, tiff All OK


def test():
    import numpy as np
    import cv2

    src = cv2.imread(
        "/Users/inf/project/face/facemarker_srv/POSE/000001/MY_000001_IEU+00_PD+00_EN_A0_D0_T0_BB_M0_R1_S0.tif")
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # RGB在opencv中存储为BGR的顺序,数据结构为一个3D的numpy.array,索引的顺序是行,列,通道:
    B = src[:, :, 0]
    G = src[:, :, 1]
    R = src[:, :, 2]
    # 灰度g=p*R+q*G+t*B（其中p=0.2989,q=0.5870,t=0.1140），于是B=(g-p*R-q*G)/t。于是我们只要保留R和G两个颜色分量，再加上灰度图g，就可以回复原来的RGB图像。
    g = src_gray[:]
    p = 0.2989;
    q = 0.5870;
    t = 0.1140
    B_new = (g - p * R - q * G) / t
    B_new = np.uint8(B_new)
    src_new = np.zeros((src.shape)).astype("uint8")
    src_new[:, :, 0] = B_new
    src_new[:, :, 1] = G
    src_new[:, :, 2] = R
    # 显示图像
    cv2.imshow("input", src)
    cv2.imshow("output", src_gray)
    cv2.imshow("result", src_new)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def restruct_pictures():
    path = a + '/pictures'
    pic_path = '/pictures_restruct'
    g = os.walk(path)
    # os.path.join(path, file_name)
    for path, dir_list, file_list in g:
        # print(path)
        g = os.walk(path)
        for _path, _dir_list, _file_list in g:
            for dir in _dir_list:

                # print(_path)
                folder = os.path.join(a,'pictures_resturct', dir)
                # print(folder)
                i = 1
                print(folder)
                if not os.path.exists(folder):  # 判断当前路径是否存在，没有则创建new文件夹
                    os.makedirs(folder)
                for filename in os.listdir(_path + '/' + dir):
                    # print(filename)
                    if os.path.splitext(filename)[-1] == ".jpg":
                        file = _path+'/'+dir+'/'+filename
                        # print(file)
                        img = cv2.imread(file)
                        img_name = os.path.join(folder, dir+'_%04d'%i + '.jpg')
                        cv2.imwrite(img_name, img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
                        # print(img_name)
                        i=i+1

# print(path)

def remove_file():
    path = a + '/pictures_restruct'
    g = os.walk('/Users/inf/project/face/facemarker_srv/pictures_resturct')
    # os.path.join(path, file_name)
    for path, dir_list, file_list in g:
        for file in file_list:
            if file.startswith('.'):
                print(file)
                print(path+'/'+file)
                os.remove(path+'/'+file)

# print(path)

def remove_dir():
    path = a+'/pictures_restruct'
    g = os.walk('/Users/inf/project/face/facemarker_srv/pictures_resturct')

    # os.path.join(path, file_name)

    for path, dir_list, file_list in g:
        # print(path)
        if re.match(r'.+学', path):
            print(path)
def rename():

    g = os.walk('/Users/inf/project/face/facemarker_srv/pictures')
    # os.path.join(path, file_name)
    for path, dir_list, file_list in g:
        # print(path)
        g = os.walk(path)
        for _path, _dir_list, _file_list in g:
            for dir in _dir_list:

                # print(_path)
                folder = os.path.join(a, 'pictures_resturct', dir)
                # print(folder)
                i = 1
                print(folder)
                if not os.path.exists(folder):  # 判断当前路径是否存在，没有则创建new文件夹
                    os.makedirs(folder)
                for filename in os.listdir(_path + '/' + dir):
                    # print(filename)
                    if os.path.splitext(filename)[-1] == ".jpg":
                        file = _path + '/' + dir + '/' + filename
                        # print(file)
                        img = cv2.imread(file)
                        img_name = os.path.join(folder, dir + '_%04d' % i + '.jpg')
                        cv2.imwrite(img_name, img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
                        # print(img_name)
                        i = i + 1

def find_mul_pic():
    g = os.walk('/Users/inf/project/face/facemarker_srv/pictures_resturct')

    # os.path.join(path, file_name)
    i = 0
    for path, dir_list, file_list in g:
        # print(path)

        file_dir = os.listdir(path)
        if(len(file_dir)>1):
            i = i+1
            print(path)
            print(len(file_dir))

    print(i)

if __name__ == '__main__':
    # restruct_pictures()
    # remove_file()
    # remove_dir()
    find_mul_pic()