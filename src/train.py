from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import cv2

import model, utils

tf.enable_eager_execution()

BATCH_SIZE = 50
LAMBDA = 10
EPOCHS = 20


### Loads the mnist training set to train the critic and the corresponding
###	random seed values necessary to train generator
def load_training_data():

	(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()

	# Normalize values between -1 and 1
	train_images = utils.mnist_normalize(train_images.astype(float))
	train_images = tf.cast(np.reshape(train_images, (-1, 28, 28, 1)), dtype=tf.float32)

	# Generate 60000 batches of length 100 random vectors with values between -1 and 1
	seeds = tf.cast(2 * np.random.rand(60000, 100) - 1, dtype=tf.float32)

	dataset = tf.data.Dataset.from_tensor_slices((train_images, seeds)).shuffle(100000).batch(BATCH_SIZE)

	return dataset

### Wasserstein GAN generator loss function
def generator_loss(generator, critic, seed):
	return tf.math.negative(tf.keras.backend.sum(critic(generator(seed))))

### Wasserstein GAN critic loss function
def critic_loss(generator, critic, seed, reals):

	fakes = generator(seed)

	# Sample uniformly along straight lines between pairs of points from data
	# distribution and generator distribution
	differences = fakes - reals
	alpha = np.random.rand(*np.shape(differences))
	interpolates = reals + alpha * differences

	# Calculate gradient penalty
	with tf.GradientTape() as gen_tape:
		estimate = critic(interpolates)
	gradients = gen_tape.gradient(estimate, critic.variables)[0]
	slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
	gradient_penalty = tf.reduce_mean((slopes-1.)**2)

	return tf.math.reduce_sum(critic(fakes)) - tf.math.reduce_sum(critic(reals)) + LAMBDA*gradient_penalty

### Used to train the generator one step
def generator_train_step(generator, critic, opt, seed):

	with tf.GradientTape() as gen_tape:
		loss = generator_loss(generator, critic, seed)

	grads = gen_tape.gradient(loss, generator.variables)
	opt.apply_gradients(zip(grads, generator.variables))

	return loss

### Used to train the critic one step
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

	sample_counter = 1

	for _ in range(0, EPOCHS):
		dataset = load_training_data()
		dataset_iter = dataset.__iter__()
		critic_loss = 0.0

		for sample in dataset_iter:
			images, seeds = sample
			# Train the generator every fifth step
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
