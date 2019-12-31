from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import cv2

import model, utils

### TODO:
### Add Comments
### Make github code nice

tf.enable_eager_execution()
tf.logging.set_verbosity(tf.logging.ERROR)

### load_training_data loads the mnist training data consisting of
###     handwritten digits and labels giving their value.
### Dimensions of returned train_images is 60000 x 28 x 28 x 1
### Dimensions of returned train_labels is 60000 x 10
def load_training_data():

	(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
	train_images = utils.mnist_normalize(train_images.astype(float))
	train_images = tf.cast(np.reshape(train_images, (-1, 28, 28, 1)), dtype=tf.float32)

	seeds = tf.cast(2 * np.random.rand(60000, 100) - 1, dtype=tf.float32)

	dataset = tf.data.Dataset.from_tensor_slices((train_images, seeds)).shuffle(100000).batch(50)

	return dataset

def generator_loss(generator, critic, seed):
	return tf.math.negative(tf.keras.backend.sum(critic(generator(seed))))

def critic_loss(generator, critic, seed, reals):
	batch_size = np.shape(seed)[0]
	line_epsilon = np.random.rand(batch_size)
	fakes = generator(seed)

	differences = fakes - reals
	alpha = np.random.rand(*np.shape(differences))
	interpolates = reals + alpha * differences

	with tf.GradientTape() as gen_tape:
		estimate = critic(interpolates)
	gradients = gen_tape.gradient(estimate, critic.variables)[0]

	slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
	gradient_penalty = tf.reduce_mean((slopes-1.)**2)

	lda = 10
	penalty = lda * gradient_penalty

	return tf.math.reduce_sum(critic(fakes)) - tf.math.reduce_sum(critic(reals)) + penalty

def generator_train_step(generator, critic, opt, seed):

	with tf.GradientTape() as gen_tape:
		loss = generator_loss(generator, critic, seed)

	grads = gen_tape.gradient(loss, generator.variables)
	opt.apply_gradients(zip(grads, generator.variables))

	return loss

def critic_train_step(generator, critic, opt, reals, seed):

	with tf.GradientTape() as gen_tape:
		loss = critic_loss(generator, critic, seed, reals)

	grads = gen_tape.gradient(loss, critic.variables)
	opt.apply_gradients(zip(grads, critic.variables))

	return loss


def main():
	generator = model.make_generator()
	critic = model.make_critic()

	gen_opt = tf.keras.optimizers.Adam(1e-4)
	critic_opt = tf.keras.optimizers.Adam(1e-4)

	checkpoint_prefix, manager, checkpoint = utils.make_checkpoint(generator, critic)
	checkpoint.restore(tf.train.latest_checkpoint(checkpoint_prefix))

	sample_seed = tf.cast(np.random.rand(1, 100), dtype=tf.float32)

	epochs = 1000
	sample_counter = 1

	for _ in range(0, epochs):
		dataset = load_training_data()
		dataset_iter = dataset.__iter__()
		critic_loss = 0.0

		for sample in dataset_iter:
			images, seeds = sample
			if sample_counter % 5 == 0:
				generator_train_step(generator, critic, gen_opt, seeds)
				print("Loss: ", critic_loss / 5)
				critic_loss = 0.0
			else:
				l = critic_train_step(generator, critic, critic_opt, images, seeds)
				critic_loss += tf.keras.backend.get_value(l)

			if sample_counter % 200 == 0:
				manager.save()
				print("Model Saved!")

			sample_counter += 1




main()
