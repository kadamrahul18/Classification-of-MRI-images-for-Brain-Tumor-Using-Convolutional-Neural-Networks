import tensorflow as tf
import datetime
import matplotlib.pyplot as plt
from PIL import Image
import numpy
from pathlib import Path
from unet import *
from augment import *
import tensorflow as tf
from IPython.display import clear_output


dataset_path = input("Enter the dataset path :")

model = unet()

sample_image_path = dataset_path + "/test_frames/test/test_frame_001.png"
sample_mask_path = dataset_path + "/test_masks/test/test_mask_001.png"

sample_image = Image.open(sample_image_path)
sample_image = numpy.array(sample_image)
sample_image_array = tf.expand_dims(sample_image, axis=-1)
sample_mask = Image.open(sample_mask_path)
sample_mask = numpy.array(sample_mask)
sample_mask_array = tf.expand_dims(sample_mask, axis=-1)


def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Input Image', 'True Mask', 'Predicted Mask']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]), cmap = 'gray')
    plt.axis('off')
  plt.show()

def create_mask(pred_mask):
  pred_mask *= 255.0
  pred_mask = tf.squeeze(pred_mask, axis=0)
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = tf.expand_dims(pred_mask, axis=-1)

  return pred_mask

def show_predictions(dataset=None, num=1):
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = model.predict(image)
      display([image[0], mask[0], create_mask(pred_mask)])
  else:
    display([sample_image_array, sample_mask_array,
             create_mask(model.predict(sample_image_array[tf.newaxis, ...]))])

class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    clear_output(wait=True)
    show_predictions()
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

STEPS_PER_EPOCH = 234 // 16 # Train_frames / batch_size
VALIDATION_STEPS = 67 // 16 # validation_frame / batch_size

weights_path = '/home/rahul/Desktop/project/weights'
Path(weights_path).mkdir(exist_ok = True)
logdir_path = '/home/rahul/Desktop/project/logs'
Path(logdir_path).mkdir(exist_ok = True)
datetime = datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
logdir = Path(logdir_path) / datetime
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

callbacks = [
    # to show samples after each epoch
    #DisplayCallback(),
    # to collect some useful metrics and visualize them in tensorboard
    tensorboard_callback,
    # if no accuracy improvements we can stop the training directly
    tf.keras.callbacks.EarlyStopping(patience=10, verbose=1),
    # to save checkpoints
    tf.keras.callbacks.ModelCheckpoint('/home/rahul/Desktop/project/weights/best_model_unet.h5', verbose=1, save_best_only=True, save_weights_only=True)
]

train_set = augmentation('train', dataset_path)
val_set = augmentation('valid', dataset_path)

model.summary()

model_history = model.fit(train_set, batch_size = 16, epochs = 20,
                          steps_per_epoch = STEPS_PER_EPOCH,
                          validation_steps = VALIDATION_STEPS,
                          validation_data = val_set,
                          callbacks = callbacks,
                          use_multiprocessing = True)

loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

epochs = range(20)

plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()
