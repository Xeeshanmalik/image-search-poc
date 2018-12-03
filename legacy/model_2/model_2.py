try:

    from keras.models import Model
    from keras.layers import *

    class Model_2:

        @staticmethod
        def Model_2(image_shape):

             # FCN ZNET

            input_layer = Input(shape=image_shape)

            # -------------------------------------------Down Sampling-------------------------------------------------

            # Output Size after First Convolutional Layer: 240 x 320 Same Padding would be equal to 1 = (240+2*(p)-3)+1
            # = if p==1 then size will be 240

            c1 = Conv2D(filters=8, kernel_size=(3, 3), activation='relu', padding='SAME')(input_layer)

            # Output Size after Second Convolutional Layer: 120 x 160 Same Padding would be again equal to 1

            c11 = Conv2D(filters=8, kernel_size=(3, 3), activation='relu', strides=(2, 2), padding='SAME')(c1)

            # Transposed Convolutional Layer c1 and c11 (Concatination of Convolution with Upsampled layer).
            # Deconvolution mostly
            # helps to smoother the structure of image and to reverse the effect of convolution on recorded data
            # Output Size by concatinating again c1 and c11 will be 240 x 320 x (64 + 8) = 72

            c12 = concatenate([Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(c11), c1], axis=-1) #240 x 320x72

            # Output Size after Third Convolutional Layer: 120 x 160 Same Padding would be equal to 1 = (240 + 2*(p) - 3) + 1
            # = if p==1 then size will be 240 with a stride of 2

            c13 = Conv2D(filters=8, kernel_size=(3, 3), activation='relu', strides=(2, 2), padding='SAME')(c12) #120 x 160

            # Output Size after Fourth Convolutional Layer: 120 x 160 Same Padding would be equal to 1 = (120+2*(p)-3)+1
            # = if p==1 then size will be 120 with a stride of 1

            c2 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='SAME')(c13) #120 x 160

            # Output Size after Fifth Convolutional Layer: 60 x 80 Same Padding would be equal to 1 = 60+2*(p)-3)+1
            # = if p==1 then size will be 120/2=60 and 160/2=80 with a stride of 2

            c21 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu', strides=(2, 2), padding='SAME')(c2) #60 x 80

            # Transposed Convolutional Layer c2 and c21 (Concatination of Convolution with Upsampled layer).
            # Deconvolution mostly helps to smoother the structure of image and to reverse the effect of convolution on
            # recorded data output size by concatinating again c21 and c2 will be 120 x 160

            c22 = concatenate([Conv2DTranspose(filters=64, kernel_size=(3,3), strides=(2, 2), padding='SAME')(c21), c2], axis=-1) #120 x 160

            # Output Size after Sixth Convolutional Layer: 60 x 80 with the stride of 2 to downsample output activations

            c23 = Conv2D(filters=8, kernel_size=(3, 3), activation='relu', strides=(2, 2), padding='SAME')(c22) #60 x 80

            # Output Size after Seventh Convolutional Layer will remain the same with stride of 1

            c3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='SAME')(c23) #60 x 80

            # Output Size after Eight Convolutional Layer will remain the same with stride of 1

            c31 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', strides=(2, 2), padding='SAME')(c3) #30 x 40

            # Transposed Convolutional Layer c2 and c21 (Concatination of Convolution with Upsampled layer).
            # Deconvolution mostly helps to smoother the structure of image and to reverse the effect of convolution on
            # recorded data output size by concatinating again c31 and c3 will be 120 x 160

            c32 = concatenate([Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(c31), c3], axis=-1) #120 x 160

            # Output Size after ninth convolutional layer will decrease to 60 x 80 with a stride of 2

            c33 = Conv2D(filters=8, kernel_size=(3, 3), activation='relu', strides=(2, 2), padding='SAME')(c32) # 60 x 80

            # Output Size after tenth convolutional layer will decrease to 30 x 40 with a stride of 2

            c4 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='SAME')(c33) # 30 x 40

            # Output Size after eleventh convolutional layer will decrease to 15 x 20 with a stride of 2

            c41 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', strides=(2, 2), padding='SAME')(c4)# 15 x 20

            # -----------------------------------------Up-Sampling------------------------------------------------------

            l = concatenate([Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(c41), c4], axis=-1) # 30 x 40

            # Transposed Convolutional Layer c2 and c21 (Concatination of Convolution with Upsampled layer).
            # Deconvolution mostly helps to smoother the structure of image and to reverse the effect of convolution on
            # recorded data output size by concatinating again c41 and c4 will be 30 x 40

            l = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='SAME')(l) #30 x 40

            # Concatinating l and c31 to make it 30 x 40 x (32+32) = 64

            l = concatenate([l, c31], axis=-1)

            # Transposed Convolutional Layer c31 and l (Concatination of Convolution with Upsampled layer).
            # Deconvolution mostly helps to smoothern the structure of image and to reverse the effect of convolution on
            # recorded data output size by concatinating again c31 and l will be 60 x 80

            l = concatenate([Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(l), c3], axis=-1) # 60 x 80

            # The output size of this layer will remain the same with the default stride of 1

            l = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='SAME')(l) # 60 x 80

            # Concatinating l and c21 to make it 120 x 160

            l = concatenate([l, c21], axis=-1)

            # Transposed Convolutional Layer c2 and l (Concatination of Convolution with Upsampled layer).
            # Deconvolution mostly helps to smoothern the structure of image and to reverse the effect of convolution on
            # recorded data output size by concatinating again c2 and l will be 120 x 160

            l = concatenate([Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(l), c2], axis=-1) #120 x 160

            # The output size of this layer will remain the same with the default stride of 1

            l = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='SAME')(l) #120 x 160

            # Concatinating l and c11 to make it 120 x 160

            l = concatenate([l, c11], axis=-1)

            # Transposed Convolutional Layer c1 and l (Concatination of Convolution with Upsampled layer).
            # Deconvolution mostly helps to smoothern the structure of image and to reverse the effect of convolution on
            # recorded data output size by concatinating again c1 and l will be 240 x 320

            l = concatenate([Conv2DTranspose(filters=16, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(l), c1], axis=-1) #240 x 320

            # The output size of this layer will remain the same with the default stride of 1

            l = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='SAME')(l) #240 x 320

            # The output size of this layer will remain the same with the default stride of 1

            l = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='SAME')(l) #240 x 320

            l = Dropout(0.3)(l)

            # l = MaxPooling2D((64,64), padding='same')(l)  # encoding layer

            # The output size of this layer will remain the same with the default stride of 1

            output_layer = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(l)

            return Model(input_layer, output_layer)


except ImportError as E:
    raise E