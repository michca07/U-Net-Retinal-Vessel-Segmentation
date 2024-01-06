import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers

class UNetBuilder:
    @staticmethod
    def build_unet(input_shape, dropout_rate, l2_penalty):
        inputs = Input(shape=input_shape)

        # Encoder
        conv1 = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(l2_penalty))(inputs)
        conv1 = BatchNormalization(axis=3)(conv1)
        conv1 = Dropout(dropout_rate)(conv1)
        conv1 = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        conv1 = BatchNormalization(axis=3)(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        conv2 = BatchNormalization(axis=3)(conv2)
        conv2 = Dropout(dropout_rate)(conv2)
        conv2 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        conv2 = BatchNormalization(axis=3)(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        conv3 = BatchNormalization(axis=3)(conv3)
        conv3 = Dropout(dropout_rate)(conv3)
        conv3 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        conv3 = BatchNormalization(axis=3)(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = BatchNormalization(axis=3)(conv4)
        conv4 = Dropout(dropout_rate)(conv4)
        conv4 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        conv4 = BatchNormalization(axis=3)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        # Bottom
        conv5 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(l2_penalty))(pool4)
        conv5 = BatchNormalization(axis=3)(conv5)
        conv5 = Dropout(dropout_rate)(conv5)
        conv5 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        conv5 = BatchNormalization(axis=3)(conv5)

        # Decoder
        up6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv5)
        merge6 = concatenate([up6, conv4])
        conv6 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        conv6 = BatchNormalization(axis=3)(conv6)
        conv6 = Dropout(dropout_rate)(conv6)
        conv6 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
        conv6 = BatchNormalization(axis=3)(conv6)

        up7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv6)
        merge7 = concatenate([up7, conv3])
        conv7 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        conv7 = BatchNormalization(axis=3)(conv7)
        conv7 = Dropout(dropout_rate)(conv7)
        conv7 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
        conv7 = BatchNormalization(axis=3)(conv7)

        up8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv7)
        merge8 = concatenate([up8, conv2])
        conv8 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = BatchNormalization(axis=3)(conv8)
        conv8 = Dropout(dropout_rate)(conv8)
        conv8 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
        conv8 = BatchNormalization(axis=3)(conv8)

        up9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv8)
        merge9 = concatenate([up9, conv1])
        conv9 = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        conv9 = BatchNormalization(axis=3)(conv9)
        conv9 = Dropout(dropout_rate)(conv9)
        conv9 = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        conv9 = BatchNormalization(axis=3)(conv9)

        # Output
        conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

        model = Model(inputs=inputs, outputs=conv10)

        return model
