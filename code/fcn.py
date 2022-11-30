# FCN model
# when tuning start with learning rate->mini_batch_size -> 
# momentum-> #hidden_units -> # learning_rate_decay -> #layers


#IMPORTING THE NEEDED MODULES
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time
import datetime
from keras.utils.vis_utils import plot_model

from utils import save_logs
from utils import calculate_metrics


#THE FCN MODEL
class Classifier_FCN:

	def __init__(self, output_directory, input_shape, nb_classes, verbose=False,build=True, batch_size=64, lr=0.009, nb_epochs=10):
		self.output_directory = output_directory
		self.nb_epochs = nb_epochs
		self.lr = lr
		self.batch_size = batch_size
		if build == True:
			self.model = self.build_model(input_shape, nb_classes)
			if(verbose==True):
				self.model.summary()
			self.verbose = verbose
			self.model.save_weights(self.output_directory+'model_init.h5')
		return

	def build_model(self, input_shape, nb_classes):
		input_layer = keras.layers.Input(input_shape)

		conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
		conv1 = keras.layers.BatchNormalization()(conv1)
		conv1 = keras.layers.Activation(activation='relu')(conv1)

		conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
		conv2 = keras.layers.BatchNormalization()(conv2)
		conv2 = keras.layers.Activation('relu')(conv2)

		conv3 = keras.layers.Conv1D(128, kernel_size=3,padding='same')(conv2)
		conv3 = keras.layers.BatchNormalization()(conv3)
		conv3 = keras.layers.Activation('relu')(conv3)

		gap_layer    = keras.layers.GlobalAveragePooling1D()(conv3)
		output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)
		model        = keras.models.Model(inputs=input_layer, outputs=output_layer)

		model.compile(loss='categorical_crossentropy', optimizer = keras.optimizers.Adam(learning_rate=self.lr),
			metrics=['accuracy'])

		# CREATING MODEL CALLBACKS
		reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)

		file_path        = self.output_directory+'best_model.h5'
		model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', save_best_only=True)

		log_dir              = self.output_directory + 'tensorboard/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
		tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, update_freq="epoch")

		self.callbacks = [reduce_lr, model_checkpoint, tensorboard_callback]

		return model 

	def fit(self, x_train, y_train, x_val, y_val,y_true):
		if not tf.test.is_gpu_available:
			print('error')
			exit()
		# x_val and y_val are only used to monitor the test loss and NOT for training
		mini_batch_size = int(min(x_train.shape[0]/10, self.batch_size))

		start_time = time.time() 

		hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=self.nb_epochs,
			verbose=self.verbose, validation_data=(x_val,y_val), callbacks=self.callbacks)

		
		duration = time.time() - start_time

		self.model.save(self.output_directory+'last_model.h5')

		model  = keras.models.load_model(self.output_directory+'best_model.h5')

		y_pred = model.predict(x_val)

		# convert the predicted from binary to integer 
		y_pred = np.argmax(y_pred , axis=1)

		save_logs(self.output_directory, hist, y_pred, y_true, duration)

		keras.backend.clear_session()

	def predict(self, x_test, y_true, x_train, y_train, y_test, return_df_metrics=True):

		# LOADING THE MODEL
		start_time = time.time()
		model_path = self.output_directory + 'best_model.h5'
		model = keras.models.load_model(model_path)

		# PREDICTING THE SAMPLES FROM THE TEST DATASET
		time1 = time.time()
		y_pred = model.predict(x_test, batch_size=self.batch_size)
		time2 = time.time()
		print("time to go through the test set:", time2 - time1)
		if return_df_metrics:
			y_pred = np.argmax(y_pred, axis=1)
			df_metrics = calculate_metrics(y_true, y_pred, 0.0)
			return df_metrics
		else:
			test_duration = time.time() - start_time
			save_test_duration(self.output_directory + 'test_duration.csv', test_duration)
			return y_pred