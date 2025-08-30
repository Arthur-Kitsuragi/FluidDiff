import FluidDiff
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

ddm = FluidDiff.DiffusionModel()

ddm.built = True
ddm.load_weights("weights.40.weights.h5")

test_images = np.load("example_data.npy")

test_images = test_images[0,:,:,:]

test_images = np.expand_dims(test_images, axis=0)

batch_tf = tf.convert_to_tensor(test_images, dtype=tf.float32)

generated_batch = ddm.generate(1, 20, batch_tf)

generated_batch = generated_batch.numpy()

img = generated_batch[0, :, :, 0]

plt.imshow(img, cmap='viridis')

plt.colorbar()

plt.show()