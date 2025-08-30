import tensorflow as tf
import math
from tensorflow.keras.layers import Layer
from tensorflow.keras import (
    layers,
    models,
    metrics,
    activations,
)

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0 = all, 1 = INFO, 2 = WARNING, 3 = ERROR

import warnings
warnings.filterwarnings("ignore")

IMAGE_SIZE = 64
BATCH_SIZE = 8
LOAD_MODEL = True
NOISE_EMBEDDING_SIZE = 128
PLOT_DIFFUSION_STEPS = 20

# optimization
EMA = 0.999
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 40

def offset_cosine_diffusion_schedule(diffusion_times):
    min_signal_rate = 0.02
    max_signal_rate = 0.95
    start_angle = tf.acos(max_signal_rate)
    end_angle = tf.acos(min_signal_rate)

    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

    signal_rates = tf.cos(diffusion_angles)
    noise_rates = tf.sin(diffusion_angles)

    return noise_rates, signal_rates

class GroupNorm(layers.Layer):
    def __init__(self, groups=4, epsilon=1e-5, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.groups = groups
        self.epsilon = epsilon

    def build(self, input_shape):
        C = input_shape[-1]
        self.G = min(self.groups, C)

        self.gamma = self.add_weight(
            shape=(1, 1, 1, C),
            initializer="ones",
            trainable=True,
            name=f"{self.name}_gamma" if self.name else "gamma"
        )
        self.beta = self.add_weight(
            shape=(1, 1, 1, C),
            initializer="zeros",
            trainable=True,
            name=f"{self.name}_beta" if self.name else "beta"
        )

    def call(self, x):
        N, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        G = self.G
        channels_per_group = C // G

        x_reshaped = tf.reshape(x, [N, H, W, G, channels_per_group])

        mean, var = tf.nn.moments(x_reshaped, axes=[1, 2, 4], keepdims=True)

        x_norm = (x_reshaped - mean) / tf.sqrt(var + self.epsilon)

        x_norm = tf.reshape(x_norm, [N, H, W, C])

        return x_norm * self.gamma + self.beta

def TimeEmbeddingMLP(embedding_dim, out_dim):
    dense1 = layers.Dense(embedding_dim, activation=activations.swish)
    dense2 = layers.Dense(out_dim)

    def apply(t_emb):
        x = dense1(t_emb)
        x = dense2(x)
        return x

    return apply

def ResNetBlock(width):
    def apply(x):
        residual = x
        x = layers.Conv2D(width, kernel_size=3, padding="same", use_bias=False)(x)
        x = GroupNorm()(x)
        x = tf.keras.layers.Activation("swish")(x)
        if residual.shape[-1] != width:
            residual = layers.Conv2D(width, kernel_size=1, padding="same", use_bias=False)(residual)
        x += residual
        return x
    return apply

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-5)
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-5)
        self.ffn1 = layers.Dense(ff_dim, activation='swish')
        self.ffn2 = layers.Dense(embed_dim) #32

    def call(self, x):
        B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]

        x_flat = tf.reshape(x, [B, H*W, C])
        x_norm = self.layernorm1(x_flat)
        attn_output = self.mha(x_norm, x_norm)
        x_flat = x_flat + attn_output

        ff = self.ffn1(x_flat)
        ff = self.ffn2(ff)
        x_flat = x_flat + ff

        x = tf.reshape(x_flat, [B, H, W, C])
        return x

def DownBlock(width,time_emb):
    def apply(x):
        x, skips = x
        x = ResNetBlock(width)(x)
        x = ResNetBlock(width)(x)
        emb = TimeEmbeddingMLP(NOISE_EMBEDDING_SIZE, width)(time_emb)
        emb = layers.Reshape((1, 1, width))(emb)
        x = layers.Add()([x, emb])
        x = TransformerBlock(embed_dim=width, num_heads=4, ff_dim=4*width)(x)
        x = layers.Conv2D(width, kernel_size=3, strides=2, padding="same")(x)
        skips.append(x)
        return x
    return apply

def UpBlock(width,time_emb):
    def apply(x):
        x, skips = x
        x = layers.Concatenate()([x, skips.pop()])
        x = ResNetBlock(width)(x)
        x = ResNetBlock(width)(x)
        emb = TimeEmbeddingMLP(NOISE_EMBEDDING_SIZE, width)(time_emb)
        emb = layers.Reshape((1, 1, width))(emb)
        x = layers.Add()([x, emb])
        x = TransformerBlock(embed_dim=width, num_heads=4, ff_dim=4*width)(x)
        x = layers.Conv2DTranspose(filters=width,kernel_size=3,strides=2,padding="same")(x)
        return x
    return apply

class SinusoidalEmbedding(Layer):
    def __init__(self, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim

    def call(self, x):
        half_dim = self.embedding_dim // 2
        frequencies = tf.exp(
            tf.linspace(tf.math.log(1.0), tf.math.log(1000.0), half_dim)
        )
        angular_speeds = 2.0 * math.pi * frequencies
        embeddings = tf.concat(
            [tf.sin(x * angular_speeds), tf.cos(x * angular_speeds)], axis=-1
        )
        return embeddings

#build flu-net
noisy_images = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 4))
x = layers.Conv2D(32, kernel_size=1)(noisy_images)
noise_variances = layers.Input(shape=(1, 1, 1))
noise_embedding = SinusoidalEmbedding(NOISE_EMBEDDING_SIZE)(noise_variances)

skips = []

x = DownBlock(32, noise_embedding)((x, skips))
x = DownBlock(64, noise_embedding)((x, skips))
x = DownBlock(96, noise_embedding)((x, skips))
x = DownBlock(128, noise_embedding)((x, skips))

x = ResNetBlock(156)(x)
x = TransformerBlock(embed_dim=156, num_heads=4, ff_dim=4*156)(x)
x = ResNetBlock(156)(x)

x = UpBlock(128, noise_embedding)((x, skips))
x = UpBlock(96, noise_embedding)((x, skips))
x = UpBlock(64, noise_embedding)((x, skips))
x = UpBlock(32, noise_embedding)((x, skips))

x = layers.Conv2D(2, kernel_size=1, kernel_initializer="zeros")(x)

flu_net = models.Model([noisy_images, noise_variances], x, name="flunet")

class DiffusionModel(models.Model):
    def __init__(self):
        super().__init__()
        self.network = flu_net
        self.ema_network = models.clone_model(self.network)
        self.diffusion_schedule = offset_cosine_diffusion_schedule

    def compile(self, **kwargs):
        super().compile(**kwargs)
        self.noise_loss_tracker = metrics.Mean(name="n_loss")

    @property
    def metrics(self):
        return [self.noise_loss_tracker]

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        if training:
            network = self.network
        else:
            network = self.ema_network
        pred_noises = network(
            [noisy_images, noise_rates**2], training=training
        )
        pred_images = (noisy_images[..., :2] - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps
        current_images = initial_noise
        last_two_channels = current_images[..., -2:]

        for step in range(diffusion_steps):
            if current_images.shape[-1] < 4:
              current_images = tf.concat([current_images, last_two_channels], axis=-1)

            diffusion_times = tf.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(
                current_images, noise_rates, signal_rates, training=False
            )
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            current_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )
        return pred_images

    def generate(self, num_images, diffusion_steps, initial_noise=None):##
        if initial_noise is None:
            initial_noise = tf.random.normal(
                shape=(num_images, IMAGE_SIZE, IMAGE_SIZE, 4)
            )
        generated_images = self.reverse_diffusion(
            initial_noise, diffusion_steps
        )
        return generated_images

    def train_step(self, images):
        noises = tf.random.normal(shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 4))

        diffusion_times = tf.random.uniform(
            shape=(BATCH_SIZE, 1, 1, 1), minval=0.0, maxval=1.0
        )

        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)

        noisy_images = tf.cast(signal_rates, tf.float32) * tf.cast(images, tf.float32) + tf.cast(noise_rates, tf.float32) * tf.cast(noises, tf.float32)

        with tf.GradientTape() as tape:
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=True
            )

            noise_loss = self.loss(noises[..., :2], pred_noises[:, :, :, :2])  # used for training

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(
            zip(gradients, self.network.trainable_weights)
        )

        self.noise_loss_tracker.update_state(noise_loss)

        for weight, ema_weight in zip(
            self.network.weights, self.ema_network.weights
        ):
            ema_weight.assign(EMA * ema_weight + (1 - EMA) * weight)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, images):#
        noises = tf.random.normal(shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 4))
        diffusion_times = tf.random.uniform(
            shape=(BATCH_SIZE, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises
        pred_noises, pred_images = self.denoise(
            noisy_images, noise_rates, signal_rates, training=False
        )
        noise_loss = self.loss(noises[..., :2], pred_noises[:, :, :, :2])
        self.noise_loss_tracker.update_state(noise_loss)

        return {m.name: m.result() for m in self.metrics}