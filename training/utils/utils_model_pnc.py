import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras import models
import logging
# import tensorflow_compression as tfc
# import tensorflow_probability as tfp
from .utils_imagenet import TailDropout1D, TailDropout

class AsymAE:
    def __init__(self, img_size=(224, 224), out_size=8):
        self.img_height, self.img_width = img_size[0], img_size[1]
        self.out_size = out_size
    
    def encoder(self):
        pass

    def decoder(self):
        pass

    def asym_ae(self):
        pass


class AsymAE_two_conv(AsymAE):
    def __init__(self, mg_size=(224, 224), out_size=8):
        super().__init__(mg_size, out_size)

#     def encoder(self):
#         encoder_input = layers.Input(shape=(224, 224, 3))
#         # Encoder
#         initializer = tf.keras.initializers.Orthogonal()
#         x = tf.keras.layers.ZeroPadding2D(8)(encoder_input)
#         x = layers.Conv2D(16, (4, 4),
#                                 strides=4,
#                                 activation="relu",
#                                 kernel_initializer=initializer,
#                                 # kernel_regularizer=self.orthogonal_reg
#                                 )(x)
#         x = tf.keras.layers.ZeroPadding2D(2)(x)
#         x = layers.Conv2D(16, (3, 3),
#                                 strides=2,
#                                 activation="relu",
#                                 padding="same",
#                                 kernel_initializer=initializer,
#                                 # kernel_regularizer=self.orthogonal_reg
#                                 )(x)
#         x = layers.Conv2D(16, (3, 3),
#                                 strides=2,
#                                 activation="relu",
#                                 padding="same",
#                                 kernel_initializer=initializer,
#                                 # kernel_regularizer=self.orthogonal_reg
#                                 )(x)
#         x = tf.keras.layers.Reshape((-1,16), input_shape=(16, 16, 16))(x)
#         encoder_model = keras.Model(encoder_input, x, name='encoder')
#         # self.encoder_output_height, self.encoder_output_width = [encoder_model.output_shape[i] for i in [1,2]]
#         encoder_model.summary()
#         self.encoder_output_height, self.encoder_output_width = [encoder_model.output_shape[i] for i in [1,2]]


#         return encoder_model

#     def decoder(self):
#         decoder_input = layers.Input(shape=(self.encoder_output_height, self.encoder_output_width))
#         x = tf.keras.layers.Reshape((16,16,16))(decoder_input)
#         x = layers.Conv2DTranspose(64, (5, 5), 
#                                     strides=2, 
#                                     activation="relu", 
#                                     padding="same",
#                                     # name="decoder_input"
#                                     )(x)

#         x = layers.Conv2DTranspose(64, (5, 5), 
#                                     strides=2, 
#                                     activation="relu", 
#                                     padding="same",
#                                     # name="decoder_input"
#                                     )(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.Conv2DTranspose(64, (9, 9), 
#                                     strides=7, 
#                                     activation="relu", 
#                                     padding="same",
#                                     # name="decoder_input"
#                                     )(x)
#         x = layers.Conv2D(16, (3, 3),
#                                 strides=2,
#                                 activation="relu",
#                                 padding="same",
#                                 )(x)
#         x = layers.BatchNormalization()(x)
#         x = layers.Conv2D(3, (3, 3),
#                                 strides=1,
#                                 activation="relu",
#                                 padding="same",
#                                 )(x)
#         x = tfp.math.clip_by_value_preserve_gradient(x, clip_value_min=0, clip_value_max=1) 

#         decoder_model = keras.Model(decoder_input, x, name='decoder')

#         # decoder_model.summary()

#         return decoder_model

    def asym_ae(self, tailDrop:bool):
        x_init = layers.Input(shape=(self.img_height, self.img_width, 3))
        x = self.encoder()(x_init)
        if tailDrop:
            logging.info("<<<<<<<<<<<<<<<<<< RAND TAIL DROP ON >>>>>>>>>>>>>>>>>>>>>>>>")
            x = TailDropout1D(func='nonequal', shape=x.shape[-2:]).dropout_uniform()(x)
        else:
            logging.info("<<<<<<<<<<<<<<<<<< RAND TAIL DROP OFFFFF >>>>>>>>>>>>>>>>>>>>>>>>")
        x = self.decoder()(x)

        autoencoder = models.Model(x_init, x)
        
        # autoencoder.summary()

        return autoencoder

class FrameCorr():
    def __init__(self, no_of_frames, out_size):
        self.no_of_frames = no_of_frames
        self.out_size = out_size
    def frame_corr(self):
        x_i = layers.Input(shape=(self.no_of_frames, None, None, self.out_size))
        separated_frames = tf.unstack(x_i, axis=1)
        dropped_out_frames = []
        for i, frame in enumerate(separated_frames):
            dropped_out_frame = TailDropout(name="Dropout_" + str(i)).dropout_uniform()(frame)
            dropped_out_frames.append(dropped_out_frame)
            if i == 0:
                concatenated = tf.add(dropped_out_frame, 0)
            else:
                concatenated = tf.add(dropped_out_frame, concatenated)
        # input = []
        # tailDropouts = []
        # for i in range(self.no_of_frames):
        #     x_i = layers.Input(shape=(self.frame_size,))
        #     input.append(x_i)
        #     tailDropouts.append(TailDropout1D(name="TailDrop" + str(i)).dropout_uniform()(x_i))
        # concatenated = tf.keras.layers.Concatenate(axis=-1)(tailDropouts)

        dense1 = tf.keras.layers.Dense(32, activation='relu')
        output1 = dense1(concatenated)

        dense2 = tf.keras.layers.Dense(self.out_size, activation='sigmoid')
        output2 = dense2(output1)

        model = models.Model(x_i, output2)

        return model

class PNC_Decoder():
    def __init__(self, prev_frames, img_size=(224, 224), out_size=10):
        self.img_height, self.img_width = img_size[0], img_size[1]
        self.out_size = out_size
        self.prev_frames = prev_frames
        # self.frame_corr = frame_corr

    def encoder(self):
        encoder_input = layers.Input(shape=(self.img_height, self.img_width, 3))
        initializer = tf.keras.initializers.Orthogonal()
        encoder_x = layers.Conv2D(16, (9, 9), 
                        strides=7, 
                        activation="relu", 
                        padding="same", 
                        kernel_initializer=initializer
                        )(encoder_input)

        encoder_x = layers.Conv2D(10, (3, 3), 
                        strides=1,
                        activation="relu", 
                        padding="same", 
                        kernel_initializer=initializer,
                        name='encoder_out'
                        )(encoder_x)

        encoder_model = keras.Model(encoder_input, encoder_x, name='encoder')
        self.encoder_output_height, self.encoder_output_width = [encoder_model.output_shape[i] for i in [1,2]]
        encoder_model.summary()

        return encoder_model

    def decoder(self, tailDrop = True):
        self.encoder()
        decoder_input = layers.Input(shape=(2, self.encoder_output_height, self.encoder_output_width, self.out_size), name='decoder_input')
        if tailDrop:
            total_dim = tf.shape(decoder_input)[-1]
            tail_len = tf.random.uniform([1, ], minval=0, maxval=total_dim, dtype=tf.int32)
            head_len = total_dim - tail_len
            mask1 = tf.concat((tf.ones([tf.shape(decoder_input)[2], tf.shape(decoder_input)[3], head_len[0]]),
                            tf.zeros((tf.shape(decoder_input)[2], tf.shape(decoder_input)[3], tail_len[0]))), axis=-1)
            mask2 = tf.concat((tf.zeros([tf.shape(decoder_input)[2], tf.shape(decoder_input)[3], head_len[0]]),
                            tf.ones((tf.shape(decoder_input)[2], tf.shape(decoder_input)[3], tail_len[0]))), axis=-1)
            
            X = decoder_input[:, 1] * mask1 + decoder_input[:, 0] * mask2
        else:
            X = decoder_input[:, 0]
        # print(X.shape)
        # print(tail_predicted.shape)
        # print(predicted_input.shape)
        decoder_x = layers.Conv2DTranspose(64, (9, 9), 
                                strides=7, 
                                activation="relu", 
                                padding="same",
                                # name="decoder_input"
                                )(X)

        decoder_x = layers.Conv2D(64, (5, 5), strides=1, activation="relu",padding="same")(decoder_x) + decoder_x
        decoder_x = layers.Conv2D(64, (5, 5), strides=1, activation="relu",padding="same")(decoder_x)
        decoder_x = layers.Conv2D(64, (5, 5), strides=1, activation="relu",padding="same")(decoder_x) + decoder_x
        decoder_out7 = layers.Conv2D(3, (3, 3),  padding="same")(decoder_x)
        decoder_out7 = tf.clip_by_value(decoder_out7, clip_value_min=0, clip_value_max=1) 

        # Autoencoderb
        decoder_model = keras.Model(decoder_input, decoder_out7,  name='decoder')
        # decoder_model.summary()

        return decoder_model

    def asym_ae(self, tailDrop:bool, encoder_trainable = True):
        x_init = layers.Input(shape=(self.img_height, self.img_width, 3))
        self.encoder_model = self.encoder()
        self.encoder_model.trainable = encoder_trainable
        x = self.encoder_model(x_init)
        if tailDrop:
            logging.info("<<<<<<<<<<<<<<<<<< RAND TAIL DROP ON >>>>>>>>>>>>>>>>>>>>>>>>")
            x = TailDropout(func='uniform').dropout_model()(x)
        else:
            logging.info("<<<<<<<<<<<<<<<<<< RAND TAIL DROP OFFFFF >>>>>>>>>>>>>>>>>>>>>>>>")
        x = self.decoder()(x)

        autoencoder = models.Model(x_init, x)

        return autoencoder
    
    def asym_ae_drop(self, tailDrop:bool, encoder_trainable = True):
        x_init = layers.Input(shape=(self.img_height, self.img_width, 3))
        self.encoder_model = self.encoder()
        self.encoder_model.trainable = encoder_trainable
        x = self.encoder_model(x_init)
        x = layers.SpatialDropout2D(0.1)(x, training=True)
        # if tailDrop:
        #     logging.info("<<<<<<<<<<<<<<<<<< RAND TAIL DROP ON >>>>>>>>>>>>>>>>>>>>>>>>")
        #     x = TailDropout(func='uniform').dropout_model()(x)
        # else:
        #     logging.info("<<<<<<<<<<<<<<<<<< RAND TAIL DROP OFFFFF >>>>>>>>>>>>>>>>>>>>>>>>")
        x = self.decoder()(x)

        autoencoder = models.Model(x_init, x)

        return autoencoder

class AsymAE_two_conv_PNC(AsymAE):
    def __init__(self, img_size=(224, 224), out_size=10):
        self.img_height, self.img_width = img_size[0], img_size[1]
        self.out_size = out_size

    def encoder(self):
        encoder_input = layers.Input(shape=(self.img_height, self.img_width, 3))
        initializer = tf.keras.initializers.Orthogonal()
        encoder_x = layers.Conv2D(16, (9, 9), 
                        strides=7, 
                        activation="relu", 
                        padding="same", 
                        kernel_initializer=initializer
                        )(encoder_input)

        encoder_x = layers.Conv2D(10, (3, 3), 
                        strides=1,
                        activation="relu", 
                        padding="same", 
                        kernel_initializer=initializer,
                        name='encoder_out'
                        )(encoder_x)

        encoder_model = keras.Model(encoder_input, encoder_x, name='encoder')
        self.encoder_output_height, self.encoder_output_width = [encoder_model.output_shape[i] for i in [1,2]]
        encoder_model.summary()

        return encoder_model

    def decoder(self):
        decoder_input = layers.Input(shape=(self.encoder_output_height, self.encoder_output_width, self.out_size), name='decoder_input')
        decoder_x = layers.Conv2DTranspose(64, (9, 9), 
                                strides=7, 
                                activation="relu", 
                                padding="same",
                                # name="decoder_input"
                                )(decoder_input)

        decoder_x = layers.Conv2D(64, (5, 5), strides=1, activation="relu",padding="same")(decoder_x) + decoder_x
        decoder_x = layers.Conv2D(64, (5, 5), strides=1, activation="relu",padding="same")(decoder_x)
        decoder_x = layers.Conv2D(64, (5, 5), strides=1, activation="relu",padding="same")(decoder_x) + decoder_x
        decoder_out7 = layers.Conv2D(3, (3, 3),  padding="same")(decoder_x)
        decoder_out7 = tf.clip_by_value(decoder_out7, clip_value_min=0, clip_value_max=1) 

        # Autoencoderb
        decoder_model = keras.Model(decoder_input, decoder_out7,  name='decoder')
        # decoder_model.summary()

        return decoder_model

    def asym_ae(self, tailDrop:bool, encoder_trainable = True):
        x_init = layers.Input(shape=(self.img_height, self.img_width, 3))
        self.encoder_model = self.encoder()
        self.encoder_model.trainable = encoder_trainable
        x = self.encoder_model(x_init)
        if tailDrop:
            logging.info("<<<<<<<<<<<<<<<<<< RAND TAIL DROP ON >>>>>>>>>>>>>>>>>>>>>>>>")
            print("x:", x)
            x = TailDropout(func='uniform', input_data=x).dropout_model()(x)
            print("x after:", x)

        else:
            logging.info("<<<<<<<<<<<<<<<<<< RAND TAIL DROP OFFFFF >>>>>>>>>>>>>>>>>>>>>>>>")
        x = self.decoder()(x)
        autoencoder = models.Model(x_init, x)

        return autoencoder
    
    def asym_ae_drop(self, tailDrop:bool, encoder_trainable = True):
        x_init = layers.Input(shape=(self.img_height, self.img_width, 3))
        self.encoder_model = self.encoder()
        self.encoder_model.trainable = encoder_trainable
        x = self.encoder_model(x_init)
        x = layers.SpatialDropout2D(0.1)(x, training=True)
        # if tailDrop:
        #     logging.info("<<<<<<<<<<<<<<<<<< RAND TAIL DROP ON >>>>>>>>>>>>>>>>>>>>>>>>")
        #     x = TailDropout(func='uniform').dropout_model()(x)
        # else:
        #     logging.info("<<<<<<<<<<<<<<<<<< RAND TAIL DROP OFFFFF >>>>>>>>>>>>>>>>>>>>>>>>")
        x = self.decoder()(x)

        autoencoder = models.Model(x_init, x)

        return autoencoder

# class AsymAE_deeper(AsymAE):
#     def __init__(self, img_size=(224, 224), out_size=10):
#         self.img_height, self.img_width = img_size[0], img_size[1]
#         self.out_size = out_size

#     def encoder(self):
#         encoder_input = layers.Input(shape=(self.img_height, self.img_width, 3))
#         initializer = tf.keras.initializers.Orthogonal()
#         encoder_x = layers.Conv2D(32, (9, 9), 
#                         strides=7, 
#                         activation="relu", 
#                         padding="same", 
#                         kernel_initializer=initializer
#                         )(encoder_input)
#         encoder_x = tfc.GDN()(encoder_x)
#         encoder_x = layers.Conv2D(32, (3, 3), 
#                         strides=1,
#                         activation="relu", 
#                         padding="same", 
#                         kernel_initializer=initializer,
#                         )(encoder_x)
#         encoder_x = tfc.GDN()(encoder_x)
#         encoder_x = layers.Conv2D(10, (3, 3), 
#                         strides=1,
#                         activation="relu", 
#                         padding="same", 
#                         kernel_initializer=initializer,
#                         name='encoder_out'
#                         )(encoder_x)

#         encoder_model = keras.Model(encoder_input, encoder_x, name='encoder')
#         self.encoder_output_height, self.encoder_output_width = [encoder_model.output_shape[i] for i in [1,2]]
#         encoder_model.summary()

#         return encoder_model

#     def decoder(self):
#         decoder_input = layers.Input(shape=(self.encoder_output_height, self.encoder_output_width, self.out_size), name='decoder_input')

#         decoder_x = layers.Conv2DTranspose(64, (9, 9), 
#                                 strides=7, 
#                                 activation="relu", 
#                                 padding="same",
#                                 # name="decoder_input"
#                                 )(decoder_input)
        
#         decoder_x = layers.Conv2D(64, (5, 5), strides=1, activation="relu",padding="same")(decoder_x) + decoder_x
#         decoder_x = tfc.GDN(inverse=True)(decoder_x)
#         decoder_x = layers.Conv2D(64, (5, 5), strides=1, activation="relu",padding="same")(decoder_x)

#         decoder_x = layers.Conv2D(64, (5, 5), strides=1, activation="relu",padding="same")(decoder_x) + decoder_x
#         decoder_x = tfc.GDN(inverse=True)(decoder_x)

#         decoder_out7 = layers.Conv2D(3, (3, 3),  padding="same")(decoder_x)
#         decoder_out7 = tf.clip_by_value(decoder_out7, clip_value_min=0, clip_value_max=1) 

#         # Autoencoderb
#         decoder_model = keras.Model(decoder_input, decoder_out7,  name='decoder')
#         # decoder_model.summary()

#         return decoder_model

#     def asym_ae(self, tailDrop:bool):
#         x_init = layers.Input(shape=(self.img_height, self.img_width, 3))
#         x = self.encoder()(x_init)
#         if tailDrop:
#             logging.info("<<<<<<<<<<<<<<<<<< RAND TAIL DROP ON >>>>>>>>>>>>>>>>>>>>>>>>")
#             x = TailDropout(func='uniform').dropout_model()(x)
#         else:
#             logging.info("<<<<<<<<<<<<<<<<<< RAND TAIL DROP OFFFFF >>>>>>>>>>>>>>>>>>>>>>>>")
#         x = self.decoder()(x)
#         autoencoder = models.Model(x_init, x)
#         return autoencoder


class AsymAE_deeper_2(AsymAE):
    def __init__(self, img_size=(224, 224), out_size=10):
        self.img_height, self.img_width = img_size[0], img_size[1]
        self.out_size = out_size

    def encoder(self):
        encoder_input = layers.Input(shape=(self.img_height, self.img_width, 3))
        initializer = tf.keras.initializers.Orthogonal()
        encoder_x = layers.Conv2D(32, (9, 9), 
                        strides=7, 
                        activation="relu", 
                        padding="same", 
                        kernel_initializer=initializer
                        )(encoder_input)
        encoder_x = tf.keras.layers.BatchNormalization()(encoder_x)
        encoder_x = layers.Conv2D(32, (3, 3), 
                        strides=1,
                        activation="relu", 
                        padding="same", 
                        kernel_initializer=initializer,
                        )(encoder_x)
        encoder_x =tf.keras.layers.BatchNormalization()(encoder_x)
        encoder_x = layers.Conv2D(10, (3, 3), 
                        strides=1,
                        activation="relu", 
                        padding="same", 
                        kernel_initializer=initializer,
                        name='encoder_out'
                        )(encoder_x)

        encoder_model = keras.Model(encoder_input, encoder_x, name='encoder')
        self.encoder_output_height, self.encoder_output_width = [encoder_model.output_shape[i] for i in [1,2]]
        encoder_model.summary()

        return encoder_model

    def decoder(self):
        decoder_input = layers.Input(shape=(self.encoder_output_height, self.encoder_output_width, self.out_size), name='decoder_input')

        decoder_x = layers.Conv2DTranspose(32, (9, 9), 
                                strides=7, 
                                activation="relu", 
                                padding="same",
                                # name="decoder_input"
                                )(decoder_input)
        
        decoder_x = layers.Conv2D(32, (5, 5), strides=1, activation="relu",padding="same")(decoder_x) + decoder_x
        decoder_x = layers.Conv2D(64, (5, 5), strides=1, activation="relu",padding="same")(decoder_x)
        decoder_x = layers.Conv2D(64, (5, 5), strides=1, activation="relu",padding="same")(decoder_x)+ decoder_x

        decoder_x = layers.Conv2D(64, (5, 5), strides=1, activation="relu",padding="same")(decoder_x)

        decoder_out7 = layers.Conv2D(3, (3, 3),  padding="same")(decoder_x)
        decoder_out7 = tf.clip_by_value(decoder_out7, clip_value_min=0, clip_value_max=1) 

        # Autoencoderb
        decoder_model = keras.Model(decoder_input, decoder_out7,  name='decoder')
        # decoder_model.summary()

        return decoder_model

    def asym_ae(self, tailDrop:bool):
        x_init = layers.Input(shape=(self.img_height, self.img_width, 3))
        x = self.encoder()(x_init)
        if tailDrop:
            logging.info("<<<<<<<<<<<<<<<<<< RAND TAIL DROP ON >>>>>>>>>>>>>>>>>>>>>>>>")
            x = TailDropout(func='uniform').dropout_model()(x)
        else:
            logging.info("<<<<<<<<<<<<<<<<<< RAND TAIL DROP OFFFFF >>>>>>>>>>>>>>>>>>>>>>>>")
        x = self.decoder()(x)
        autoencoder = models.Model(x_init, x)
        return autoencoder

