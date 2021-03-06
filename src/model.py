import tensorflow as tf

### Generator receives a batch of vectors of length 100 and returns
###	a batch of 28 x 28 x 1 tensors
def make_generator():

	##Returns 7 x 7 x 64 tensor from random vector of length 100
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

	## Returns 28 x 28 x 64 tensor
	model.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=1, padding='same', use_bias=False))
	model.add(tf.keras.layers.LeakyReLU())
	model.add(tf.keras.layers.BatchNormalization())

	## Returns 28 x 28 x 1 tensor
	## Tanh activation used to renormalize image values to (-1, 1)
	model.add(tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=1, padding='same', use_bias=False))
	model.add(tf.keras.layers.Activation(tf.keras.activations.tanh))

	return model


### Critic receives a batch of 28 x 28 x 1 matrices and returns a batch of values
### representing the "quality" of each sample
###
### Note:
### 	No batch normalization used in critic model
###		No activation used in last layer of model
def make_critic():
	model = tf.keras.Sequential()

	model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=2, padding='same', input_shape=(28, 28, 1), use_bias=False))
	model.add(tf.keras.layers.LeakyReLU())

	model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=2, padding='same', use_bias=False))
	model.add(tf.keras.layers.LeakyReLU())

	model.add(tf.keras.layers.Conv2D(64, (5, 5), strides=2, padding='same', use_bias=False))
	model.add(tf.keras.layers.LeakyReLU())

	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(1))

	return model
