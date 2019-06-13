from PIL import Image
import os, re

inputFileFolder = "/Users/inf/project/face/facemarker_srv/POSE_160_RENAME/"

path = r"/Users/inf/project/face/facemarker_srv/POSE_160_RENAME/"
os.chdir(path)
foldernames = os.listdir(os.getcwd())
print
foldernames

f = open("/Users/inf/project/face/facemarker_srv/POSE_pairs2.txt", 'a')
name_number = {}
foldername_list = []
for foldername in foldernames:
    if not foldername.startswith('.'):
        os.chdir(inputFileFolder + foldername)
        filenames = os.listdir(os.getcwd())

        number_list = []
        # print foldername
        for filename in filenames:
            print(filename)
            name, number = filename.split("_", 1)
            number = number.rstrip(".jpg")
            number = str(number).strip("000")
            number_list.append(number)
            number_list.sort()
        count = len(number_list)
        for i in range(0, count - 1):
            for j in range(i + 1, count):
                # 对一个文件夹下的图片进行相似排列，如写入一行 jianwenmama  1  2
                lineStr = foldername + '  ' + number_list[i] + '  ' + number_list[j]
                f.write(lineStr)
                f.write('\n')
        # print number_list
        name_number[foldername] = number_list
        foldername_list.append(foldername)
    #    print name_number
        count_foldername = len(foldername_list)

for k in range(0, count_foldername):
    # print foldername_list[k]
    one_list = name_number.get(foldername_list[k])  # print one_list
    for i in range(0, len(one_list)):
        for j in range(0, count_foldername):
            if j == k:
                continue
            next_list = name_number.get(foldername_list[j])
            print("foldername[%d]"%j,foldername_list[j])

            print("next=",next_list)

            # 对一个文件夹下的图片与其他文件夹下的图片1进行不相似排列，如写入一行 yt  5  hcz   1
            # 由于如果对一个文件夹下的图片与其他文件夹下的每张图片都进行不相似排列，会太多不相似的测试样本组合，所以这里只对其他文件夹下的图片1比较。

            negative_linestr = foldername_list[k] + "  " + str(i + 1) + "  " + foldername_list[j] + "   " + next_list[0]

            print('negative_linestr=')
            print(negative_linestr)
            f.write(negative_linestr)
            f.write('\n')
f.close()

