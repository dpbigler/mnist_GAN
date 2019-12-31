import tensorflow as tf

def make_generator():
	## Returns 7 x 7 x 64 tensor from random vector of length 100
	model = tf.keras.Sequential()
	model.add(tf.keras.layers.Dense(49, input_shape=(100,)))
	model.add(tf.keras.layers.Reshape((7, 7, 1)))
	model.add(tf.keras.layers.Activation(tf.keras.activations.tanh))
	model.add(tf.keras.layers.BatchNormalization())

	## Returns 14 x 14 x 64 tensor
	model.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=2, padding='same', use_bias=False))
	model.add(tf.keras.layers.LeakyReLU())
	model.add(tf.keras.layers.BatchNormalization())

	## Returns 28 x 28 x 64 tensor
	model.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=2, padding='same', use_bias=False))
	model.add(tf.keras.layers.LeakyReLU())
	model.add(tf.keras.layers.BatchNormalization())

	model.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=1, padding='same', use_bias=False))
	model.add(tf.keras.layers.LeakyReLU())
	model.add(tf.keras.layers.BatchNormalization())

	## Returns 28 x 28 x 1 tensor
	model.add(tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=1, padding='same', use_bias=False))
	model.add(tf.keras.layers.Activation(tf.keras.activations.tanh))

	return model

def make_critic():
	## Critic receives a batch of 28 x 28 x 1 matrices
	model = tf.keras.Sequential()

	## Returns 18 x 28 x 64 tensor
	model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=2, padding='same', input_shape=(28, 28, 1), use_bias=False))
	model.add(tf.keras.layers.LeakyReLU())

	model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=2, padding='same', use_bias=False))
	model.add(tf.keras.layers.LeakyReLU())

	model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=2, padding='same', use_bias=False))
	model.add(tf.keras.layers.LeakyReLU())

	## Returns DCGAN score
	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(1))

	return model
