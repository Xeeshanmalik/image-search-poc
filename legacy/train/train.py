try:

    import model_1.model_1 as M
    from model_1.model_1 import Model_1
    from model_2.model_2 import Model_2
    from model_3.model_3 import Model_3
    import numpy as np
    from keras.optimizers import RMSprop
    import time


    class Train:

        def __train_model_1__(self):

            model_1 = Model_1.Model_1(self.input_shape)
            model_1.compile(loss='mean_squared_error', optimizer='adam')
            self.output_image = self.input_images_model_1[:, :, :, 0:1]
            model_1.fit(self.input_images_model_1, self.output_image, self.batch_size, self.epochs, verbose=1)
            model_1.save(self.model_1_save_path + 'model_1.h5')

            return model_1

        def __train_model_2__(self):

            model_2 = Model_2.Model_2(self.input_shape)
            model_2.compile(loss='mean_squared_error', optimizer=RMSprop(0.00005))
            self.output_image = self.input_images_model_2[:, :, :, 0:1]
            model_2.fit(self.input_images_model_2, self.output_image, self.batch_size, self.epochs, verbose=1)
            model_2.save(self.model_2_save_path + 'model_2.h5')

            return model_2

        def __train_model_3__(self):

            model_3 = Model_3.Model_3(self.input_shape_3)
            model_3.compile(loss='mean_squared_error', optimizer='adam')
            self.output_image = self.input_images_model_3[:, :, :, 0:1]
            model_3.fit(self.input_images_model_3, self.output_image, self.batch_size, self.epochs, verbose=1)
            model_3.save(self.model_3_save_path + 'model_3.h5')

            return model_3

        def __init__(self, input_images_model_1, input_images_model_2, input_images_model_3, input_shape, input_shape_3 ,batch_size, epochs, model_1_save_path, model_2_save_path,model_3_save_path):

            self.input_images_model_1 = input_images_model_1
            self.input_images_model_2 = input_images_model_2
            self.input_images_model_3 = input_images_model_3
            self.input_shape = input_shape
            self.input_shape_3 = input_shape_3
            self.batch_size = batch_size
            self.epochs = epochs
            self.model_1_save_path = model_1_save_path
            self.model_2_save_path = model_2_save_path
            self.model_3_save_path = model_3_save_path

except ImportError as E:
    raise E