#coding : utf-8
import shutil
import os
lin = []
img_path = "train"
copy_to_path = '100'

for line in open("100.txt"):
    lin = line.split()
    #print(lin[0])
    #for img in lin[0]:
    shutil.copy(os.path.join(img_path, lin[0]), os.path.join(copy_to_path, lin[0]))  # 图片复制到另一个文件夹
    os.remove(os.path.join(img_path, lin[0]))  # 并删除原有文件
