import cv2
import os, fnmatch
import imgaug as ia
import numpy as np
from imgaug import augmenters as iaa

img_extension = '.jpg'
source_path = 'training_data/'
augumented_path = 'augumented_data/'

seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Crop(percent=(0, 0.1)),
    iaa.Sometimes(0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    iaa.ContrastNormalization((0.75, 1.5)),
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    )
], random_order=True)

def load_batch(breed_dir):
    breed_images = os.listdir(breed_dir)
    list = []
    for file in breed_images:
        print(file)
        if fnmatch.fnmatch(file, '*'+img_extension):
            image = cv2.imread(breed_dir+'/'+file)
            list.append(image)
    return list

def make_dir_if_not_exists(directory):
    if not(os.path.exists(directory)):
        os.makedirs(directory)

def augument_images():
    breed_dirs = os.listdir(source_path)
    for breed_dir in breed_dirs:
        if not os.path.isfile(breed_dir):
            breed_images = load_batch(source_path+breed_dir)
            images_aug =[]
            for i in range(0, 1):
                images_aug.extend(seq.augment_images(breed_images))
                print("images: ", len(breed_images))
                print("augumented images: ", len(images_aug))

            images_aug.extend(breed_images)

            aug_breed_path = augumented_path+breed_dir+'/'
            aug_breed_file_name = breed_dir.lower().replace(" ", "")
            make_dir_if_not_exists(aug_breed_path)

            for i in range(0, len(images_aug)):
                cv2.imwrite(aug_breed_path+str(i)+'_'+aug_breed_file_name+img_extension, images_aug[i])

augument_images()
