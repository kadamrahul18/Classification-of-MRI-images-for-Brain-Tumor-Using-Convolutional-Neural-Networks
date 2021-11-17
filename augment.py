import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
#from sklearn.preprocessing import LabelEncoder
import albumentations as A



# define augmentation methods, p = probability
def get_augmentation():
    transform = [
        A.OneOf([

        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=(0,90), p=0.5),
        A.ShiftScaleRotate(shift_limit = (0,0.1), rotate_limit = (0,0), scale_limit = (0,0), p=0.5), #only shift
        A.Transpose(p=0.5)

        ],p=1,
        )
    ]
    return A.Compose(transform)

'''
def mask_image_encoding(data):
    mask_encoder = LabelEncoder()
    h, w = data.shape
    data_reshaped = data.reshape(-1, 1) #reshaping mask image 2d array to 1d vector for labelEncoder() to work
    data_reshaped_encoded = mask_encoder.fit_transform(data_reshaped)
    data_encoded_original_shape = data_reshaped_encoded.reshape(h, w) #reshaping back the mask image to original 2d array
    return data_encoded_original_shape
'''
# classes for data loading and preprocessing
class Dataset:

    CLASSES = ['background','non-enhancing','edema','enhancing']

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
    ):
        self.image_ids = sorted(os.listdir(images_dir))
        self.mask_ids = sorted(os.listdir(masks_dir))
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.image_ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.mask_ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i], 0) # 0 = grayscale
        mask = cv2.imread(self.masks_fps[i], 0)
        mask = np.where(mask == 4, 3, mask) # replace all elements with value 4 to 3
        image = cv2.resize(image, (256,256), interpolation = cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (256,256), interpolation = cv2.INTER_NEAREST)

        #one-hot encoding
        mask = to_categorical(mask, num_classes=4)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']


        image = image/255.0 #normalize input image
        image = np.expand_dims(image, axis=-1)
        return image, mask

    def __len__(self):
        return len(self.image_ids)


class Dataloader(tf.keras.utils.Sequence):

    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):

        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]

        return tuple(batch)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)
