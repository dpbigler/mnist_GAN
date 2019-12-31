from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import model, utils

tf.enable_eager_execution()

def load_testing_data(batch_size):
	seeds = tf.cast(2 * np.random.rand(batch_size, 100) - 1, dtype=tf.float32)
	return seeds

### Generate n test samples. Samples are stored in "generated_images"
def generate(n):
	data = load_testing_data(n)
	generator = model.make_generator()
	critic = model.make_critic()

	checkpoint_prefix, manager, checkpoint = utils.make_checkpoint(generator, critic)
	checkpoint.restore(tf.train.latest_checkpoint(checkpoint_prefix))

	for i in range(0, 10):
		img = generator(np.reshape(data[i, :], (1, 100)))
		utils.save_tensor_as_png(img, i)
