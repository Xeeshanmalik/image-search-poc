from keras.models import Model
from keras.layers import *
from keras.layers.normalization import BatchNormalization


class autoencoder:

    def __init__(self, image_shape, classes):
        self.image_shape = image_shape
        self.classes = classes
        self.input_layer = self.image_shape

        self.layer_1 = None
        self.layer_2 = None
        self.layer_3 = None
        self.layer_4 = None
        self.layer_5 = None
        self.layer_6 = None
        self.layer_7 = None
        self.layer_8 = None
        self.layer_9 = None
        self.layer_10 = None
        self.layer_11 = None
        self.layer_12 = None
        self.layer_13 = None
        self.layer_14 = None
        self.layer_15 = None
        self.layer_16 = None
        self.layer_17 = None
        self.layer_18 = None
        self.layer_19 = None
        self.layer_20 = None

        self.concat_l1_l2 = None
        self.concat_l4_l5 = None
        self.concat_l7_l8 = None
        self.concat_l10_l11 = None
        self.concat_l7_l13 = None
        self.concat_l4_l15 = None
        self.layer_10_bn = None
        self.l_dropout = None
        self.decoder = None
        self.encoder = None

        self.stack_one_encoder = None
        self.stack_one_decoder = None

        self.stack_two_encoder = None
        self.stack_two_decoder = None

        self.max_pooled = None
        self.flatten = None

        self.Dense_1 = None
        self.Dense_2 = None

        self.output_layer = None

        self.model()

    def model(self):
        prefix_for_stack_1 = 'stack_1_'
        self.input_layer = Input(shape=self.image_shape)
        self.stack_one_encoder = self.encoded(self.input_layer, prefix_for_stack_1)

        # self.stack_one_decoder = self.decoded(self.stack_one_encoder, prefix_for_stack_1)

        # The layers are tuned now just for the time being
        # commented for the time being will enable it after few iterations
        # prefix_for_stack_2 = 'stack_2_'
        # self.stack_two_encoder = self.encoded(self.stack_one_decoder, prefix_for_stack_2 )
        # self.stack_two_decoder = self.decoded(self.stack_two_encoder, prefix_for_stack_2 )

        self.max_pooled = MaxPooling2D((2, 2), padding='same',
                                       name=prefix_for_stack_1 + 'max_pooled')(self.stack_one_encoder)
        self.flatten = Flatten(name=prefix_for_stack_1 + 'flatten')(self.max_pooled)

        self.Dense_1 = Dense(1024, activation='relu', name=prefix_for_stack_1 + 'dense_1')(self.flatten)
        self.Dense_2 = Dense(1024, activation='relu', name=prefix_for_stack_1 + 'dense_2')(self.Dense_1)

        self.output_layer = Dense(self.classes, activation='softmax',
                                  name=prefix_for_stack_1 + 'predictions')(self.Dense_2)

        return Model(self.input_layer, self.output_layer)

    def encoded(self, input_layer, prefix):
        self.layer_1 = Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='SAME',
                              name=prefix + 'layer_1')(input_layer)
        self.layer_2 = Conv2D(filters=8, kernel_size=(3, 3), activation='relu', strides=(2, 2),
                              padding='SAME', name=prefix + 'layer_2')(self.layer_1)
        self.concat_l1_l2 = concatenate([Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2),
                                                         padding='SAME',
                                                         name=prefix + 'concat_l1_l2')(self.layer_2),
                                         self.layer_1], axis=-1)
        self.layer_3 = Conv2D(filters=8, kernel_size=(3, 3), activation='relu', strides=(2, 2),
                              padding='SAME', name=prefix + 'layer_3')(self.concat_l1_l2)
        self.layer_4 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='SAME',
                              name=prefix + 'layer_4')(self.layer_3)
        self.layer_5 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', strides=(2, 2),
                              padding='SAME', name=prefix + 'layer_5')(self.layer_4)
        self.concat_l4_l5 = concatenate([Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2),
                                                         padding='SAME', name=prefix + 'concat_l4_l5')(
            self.layer_5),
            self.layer_4], axis=-1)
        self.layer_6 = Conv2D(filters=8, kernel_size=(3, 3), activation='relu',
                              strides=(2, 2), padding='SAME', name=prefix + 'layer_6')(self.concat_l4_l5)
        self.layer_7 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
                              padding='SAME', name=prefix + 'layer_7')(self.layer_6)
        self.layer_8 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', strides=(2, 2),
                              padding='SAME',
                              name=prefix + 'layer_8')(self.layer_7)
        self.concat_l7_l8 = concatenate([Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2),
                                                         padding='SAME', name=prefix + 'concat_l7_l8')(
            self.layer_8),
            self.layer_7], axis=-1)
        self.layer_9 = Conv2D(filters=8, kernel_size=(3, 3), activation='relu',
                              strides=(2, 2), padding='SAME', name=prefix + 'layer_9')(self.concat_l7_l8)
        self.layer_10 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
                               padding='SAME', name=prefix + 'layer_10')(self.layer_9)
        self.layer_10_bn = BatchNormalization()(self.layer_10)

        self.encoder = Conv2D(filters=64, kernel_size=(3, 3), activation='sigmoid',
                              strides=(2, 2), padding='SAME', name=prefix + 'encoded')(self.layer_10_bn)

        # layer_12 = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='SAME', activation='relu',
        #           name='layer_12')(layer_11) # 4 x 4
        # concat_l11_l12 = concatenate([Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='SAME',
        #                                              name='concat_l11_l12')(layer_12), layer_11], axis=-1) # 8 x 8
        # layer_13 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
        #           strides=(2, 2), padding='SAME', name='layer_13')(concat_l11_l12)
        # layer_13_bn = BatchNormalization()(layer_13) # 8 x 8
        # layer_14 = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='SAME', activation='sigmoid',
        #            name='layer_14')(layer_13_bn)
        # concat_l13_l14 = concatenate([Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='SAME',
        #                                              name='concat_l13_l14')(layer_14), layer_13], axis=-1) # 8 x 8

        return self.encoder

    def decoded(self, encoder, prefix):
        self.concat_l10_l11 = concatenate(
            [Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='SAME',
                             name=prefix + 'concat_l10_l11')(encoder), self.layer_10], axis=-1)
        self.layer_12 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='SAME',
                               name=prefix + 'layer_12')(
            self.concat_l10_l11)
        self.layer_13 = concatenate([self.layer_12, self.layer_8], axis=-1, name=prefix + 'layer_13')
        self.concat_l7_l13 = concatenate(
            [Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='SAME',
                             name=prefix + 'concat_l7_l13')(self.layer_13), self.layer_7],
            axis=-1)
        self.layer_14 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='SAME',
                               name=prefix + 'layer_14')(self.concat_l7_l13)
        self.layer_15 = concatenate([self.layer_14, self.layer_5], axis=-1, name=prefix + 'layer_15')
        self.concat_l4_l15 = concatenate([Conv2DTranspose(filters=32, kernel_size=(3, 3),
                                                          strides=(2, 2), padding='SAME',
                                                          name=prefix + 'concat_l4_15')(self.layer_15),
                                          self.layer_4],
                                         axis=-1)
        self.layer_16 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='SAME',
                               name=prefix + 'layer_16')(self.concat_l4_l15)
        self.layer_17 = concatenate([self.layer_16, self.layer_2], axis=-1, name=prefix + 'layer_17')
        self.layer_18 = concatenate([Conv2DTranspose(filters=16, kernel_size=(3, 3),
                                                     strides=(2, 2), padding='SAME',
                                                     name=prefix + 'layer_18')(self.layer_17), self.layer_1],
                                    axis=-1)
        self.layer_19 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='SAME',
                               name=prefix + 'layer_19')(self.layer_18)
        self.layer_20 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='SAME',
                               name=prefix + 'layer_20')(self.layer_19)
        self.l_dropout = Dropout(0.3)(self.layer_20)

        self.decoder = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid',
                              name=prefix + 'decoder')(self.l_dropout)
        return self.decoder
