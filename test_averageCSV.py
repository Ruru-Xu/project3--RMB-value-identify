from keras.models import load_model
import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

from keras.utils.training_utils import multi_gpu_model

img_width = 299
img_height = 299
batch_size = 32
nbr_test_samples = 1000
nbr_augmentation = 5

FishNames = ['0.1','0.2','0.5','1','2','5','10','50','100']

root_path = '/home/imc/XR/temp/MobileNetV3-Pytorch'
weights_path = os.path.join(root_path, 'RMB_value_identify/output_weights/weights.h5')

test_data_dir = os.path.join(root_path, 'datasets/RMB_data/test_RMB/')

# test data generator for prediction
test_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True)


print('Loading model and weights from training process ...')
InceptionV3_model = load_model(weights_path)

InceptionV3_model = multi_gpu_model(InceptionV3_model, gpus=4) # ����ʹ��4��gpu���þ����ģ��compile֮ǰ

for idx in range(nbr_augmentation):
    print('{}th augmentation for testing ...'.format(idx))
    random_seed = np.random.random_integers(0, 100000)

    test_generator = test_datagen.flow_from_directory(
            test_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            shuffle = False, # Important !!!
            seed = random_seed,
            classes = None,
            class_mode = None)

    test_image_list = test_generator.filenames
    #print(test_image_list)
    #print('image_list: {}'.format(test_image_list[:10]))
    print('Begin to predict for testing data ...')
    if idx == 0:
        predictions = InceptionV3_model.predict_generator(test_generator, nbr_test_samples)
    else:
        predictions += InceptionV3_model.predict_generator(test_generator, nbr_test_samples)

predictions /= nbr_augmentation

print('Begin to write submission file ..')
f_submit = open(os.path.join(root_path, 'submit-2019052102.csv'), 'w')
f_submit.write('name,0.1,0.2,0.5,1,2,5,10,50,100\n')
for i, image_name in enumerate(test_image_list):
    pred = ['%.6f' % p for p in predictions[i, :]]
    if i % 100 == 0:
        print('{} / {}'.format(i, nbr_test_samples))
    f_submit.write('%s,%s\n' % (os.path.basename(image_name), ','.join(pred)))

f_submit.close()

print('Submission file successfully generated!')
