import os
import numpy as np
import shutil

np.random.seed(2016)

root_train = '/home/imc/XR/temp/MobileNetV3-Pytorch/datasets/RMB_data/train_split'
root_val = '/home/imc/XR/temp/MobileNetV3-Pytorch/datasets/RMB_data/val_split'

root_total = '/home/imc/XR/temp/MobileNetV3-Pytorch/datasets/RMB_data/train'

RMBlabels = ['0.1','0.2','0.5','1','2','5','10','50','100']

nbr_train_samples = 0
nbr_val_samples = 0

# Training proportion
split_proportion = 0.8

for RMBlabel in RMBlabels:
    if RMBlabel not in os.listdir(root_train):
        os.mkdir(os.path.join(root_train, RMBlabel))

    total_images = os.listdir(os.path.join(root_total, RMBlabel))

    nbr_train = int(len(total_images) * split_proportion)

    np.random.shuffle(total_images)

    train_images = total_images[:nbr_train]

    val_images = total_images[nbr_train:]

    for img in train_images:
        source = os.path.join(root_total, RMBlabel, img)
        target = os.path.join(root_train, RMBlabel, img)
        shutil.copy(source, target)
        nbr_train_samples += 1

    if RMBlabel not in os.listdir(root_val):
        os.mkdir(os.path.join(root_val, RMBlabel))

    for img in val_images:
        source = os.path.join(root_total, RMBlabel, img)
        target = os.path.join(root_val, RMBlabel, img)
        shutil.copy(source, target)
        nbr_val_samples += 1

print('Finish splitting train and val images!')
print('# training samples: {}, # val samples: {}'.format(nbr_train_samples, nbr_val_samples))
