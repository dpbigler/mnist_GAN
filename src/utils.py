from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import os, cv2

## GIF/IMage Making Here!

def make_checkpoint(generator, critic):
	checkpoint_prefix = '../training_checkpoints'
	checkpoint = tf.train.Checkpoint(
			generator=generator,
			critic=critic
	)
	manager = tf.contrib.checkpoint.CheckpointManager(
		checkpoint, directory=checkpoint_prefix, max_to_keep=5)

	return checkpoint_prefix, manager, checkpoint

def save_tensor_as_png(tensor, version):
    img = np.floor(mnist_unnormalize(tensor))
    img = np.reshape(img, (28, 28)).astype(int)
    cv2.imwrite("../generated_images/" + str(version) + '.png', img)

def mnist_normalize(tensor):
    return (tensor - 128) / 128

def mnist_unnormalize(tensor):
    return 128 * tensor + 128
