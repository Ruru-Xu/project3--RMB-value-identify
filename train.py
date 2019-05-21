# coding:utf-8
from keras.applications.inception_v3 import InceptionV3
import os
from keras.layers import Flatten, Dense, AveragePooling2D
from keras.models import Model
from keras.optimizers import RMSprop, SGD
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

from sklearn.preprocessing import LabelBinarizer

from keras.utils.training_utils import multi_gpu_model

learning_rate = 0.0001
img_width = 299
img_height = 299
nbr_train_samples = 3019
nbr_validation_samples = 758
nbr_epochs = 25
batch_size = 32

train_data_dir = '/home/imc/XR/temp/MobileNetV3-Pytorch/datasets/RMB_data/train_split'
val_data_dir = '/home/imc/XR/temp/MobileNetV3-Pytorch/datasets/RMB_data/val_split'

labelNames = ['0.1','0.2','0.5','1','2','5','10','50','100']
#lb = LabelBinarizer()
#labelNames  = lb.fit_transform(labelNames ) # transfer label to binary value
#print(labelNames .shape)

print('Loading InceptionV3 Weights ...')
InceptionV3_notop = InceptionV3(include_top=False, weights='imagenet',
                    input_tensor=None, input_shape=(299, 299, 3))
# Note that the preprocessing of InceptionV3 is:
# (x / 255 - 0.5) x 2

print('Adding Average Pooling Layer and Softmax Output Layer ...')
output = InceptionV3_notop.get_layer(index = -1).output  # Shape: (8, 8, 2048)
output = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(output)
output = Flatten(name='flatten')(output)
output = Dense(9, activation='softmax', name='predictions')(output)

InceptionV3_model = Model(InceptionV3_notop.input, output)
#InceptionV3_model.summary()

InceptionV3_model = multi_gpu_model(InceptionV3_model, gpus=4) # ����ʹ��4��gpu���þ����ģ��compile֮ǰ


optimizer = SGD(lr = learning_rate, momentum = 0.9, decay = 0.0, nesterov = True)
InceptionV3_model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

# autosave best Model
best_model_file = "output_weights/weights.h5"
best_model = ModelCheckpoint(best_model_file, monitor='val_acc', verbose = 1, save_best_only = True)

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        rotation_range=10.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)

# this is the augmentation configuration we will use for validation:
# only rescaling
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        shuffle = True,
        # save_to_dir = '/home/imc/XR/temp/MobileNetV3-Pytorch/RMB_value_identify/data/visualization',
        # save_prefix = 'aug',
        classes = labelNames,
        class_mode = 'categorical')

validation_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle = True,
        #save_to_dir = '/home/imc/XR/temp/MobileNetV3-Pytorch/RMB_value_identify/data/visulization',
        #save_prefix = 'aug',
        classes = labelNames,
        class_mode = 'categorical')

InceptionV3_model.fit_generator(
        train_generator,
        samples_per_epoch = nbr_train_samples,
        nb_epoch = nbr_epochs,
        validation_data = validation_generator,
        nb_val_samples = nbr_validation_samples,
        callbacks = [best_model])
