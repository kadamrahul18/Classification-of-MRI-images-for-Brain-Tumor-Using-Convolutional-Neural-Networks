import tensorflow as tf
from tensorflow.keras.layers import Activation, Concatenate, Conv2D, Conv2DTranspose, Input, MaxPooling2D
from tensorflow.keras.optimizers import Adam


def dice_coefficient(y_true, y_pred, smooth: float = 1e-6):
    """Dice metric for one-hot encoded masks."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    numerator = 2.0 * tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    denominator = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3])
    dice = (numerator + smooth) / (denominator + smooth)
    return tf.reduce_mean(dice)


def soft_dice_loss(y_true, y_pred, smooth: float = 1e-6):
    dice = dice_coefficient(y_true, y_pred, smooth)
    return 1.0 - dice


_cce = tf.keras.losses.CategoricalCrossentropy()


def combined_cce_dice_loss(y_true, y_pred):
    """Categorical crossentropy + soft dice to sharpen boundaries."""
    return _cce(y_true, y_pred) + soft_dice_loss(y_true, y_pred)


def build_unet(
    input_size=(256, 256, 1),
    num_classes: int = 4,
    base_filters: int = 32,
    learning_rate: float = 1e-4,
) -> tf.keras.Model:
    initializer = "he_normal"

    inputs = Input(shape=input_size)

    conv1 = Conv2D(base_filters, (3, 3), activation="relu", padding="same", kernel_initializer=initializer)(inputs)
    conv1 = Conv2D(base_filters, (3, 3), activation="relu", padding="same", kernel_initializer=initializer)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(base_filters * 2, (3, 3), activation="relu", padding="same", kernel_initializer=initializer)(pool1)
    conv2 = Conv2D(base_filters * 2, (3, 3), activation="relu", padding="same", kernel_initializer=initializer)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(base_filters * 4, (3, 3), activation="relu", padding="same", kernel_initializer=initializer)(pool2)
    conv3 = Conv2D(base_filters * 4, (3, 3), activation="relu", padding="same", kernel_initializer=initializer)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(base_filters * 8, (3, 3), activation="relu", padding="same", kernel_initializer=initializer)(pool3)
    conv4 = Conv2D(base_filters * 8, (3, 3), activation="relu", padding="same", kernel_initializer=initializer)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(base_filters * 16, (3, 3), activation="relu", padding="same", kernel_initializer=initializer)(pool4)
    conv5 = Conv2D(base_filters * 16, (3, 3), activation="relu", padding="same", kernel_initializer=initializer)(conv5)

    up6 = Concatenate(axis=3)([
        Conv2DTranspose(base_filters * 8, (2, 2), strides=(2, 2), padding="same", kernel_initializer=initializer)(conv5),
        conv4,
    ])
    conv6 = Conv2D(base_filters * 8, (3, 3), activation="relu", padding="same", kernel_initializer=initializer)(up6)
    conv6 = Conv2D(base_filters * 8, (3, 3), activation="relu", padding="same", kernel_initializer=initializer)(conv6)

    up7 = Concatenate(axis=3)([
        Conv2DTranspose(base_filters * 4, (2, 2), strides=(2, 2), padding="same", kernel_initializer=initializer)(conv6),
        conv3,
    ])
    conv7 = Conv2D(base_filters * 4, (3, 3), activation="relu", padding="same", kernel_initializer=initializer)(up7)
    conv7 = Conv2D(base_filters * 4, (3, 3), activation="relu", padding="same", kernel_initializer=initializer)(conv7)

    up8 = Concatenate(axis=3)([
        Conv2DTranspose(base_filters * 2, (2, 2), strides=(2, 2), padding="same", kernel_initializer=initializer)(conv7),
        conv2,
    ])
    conv8 = Conv2D(base_filters * 2, (3, 3), activation="relu", padding="same", kernel_initializer=initializer)(up8)
    conv8 = Conv2D(base_filters * 2, (3, 3), activation="relu", padding="same", kernel_initializer=initializer)(conv8)

    up9 = Concatenate(axis=3)([
        Conv2DTranspose(base_filters, (2, 2), strides=(2, 2), padding="same", kernel_initializer=initializer)(conv8),
        conv1,
    ])
    conv9 = Conv2D(base_filters, (3, 3), activation="relu", padding="same", kernel_initializer=initializer)(up9)
    conv9 = Conv2D(base_filters, (3, 3), activation="relu", padding="same", kernel_initializer=initializer)(conv9)

    conv10 = Conv2D(num_classes, (1, 1), activation="relu", kernel_initializer=initializer)(conv9)
    outputs = Activation("softmax")(conv10)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=combined_cce_dice_loss,
        metrics=[dice_coefficient, "accuracy"],
    )
    return model
