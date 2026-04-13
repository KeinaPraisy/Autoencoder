# Autoencoder-Based Image Denoising System

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

# Dataset Preparation
def load_and_prepare_data():
    (x_train, _), (x_test, _) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    noise_factor = 0.5
    noisy_x_train = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    noisy_x_test = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

    noisy_x_train = np.clip(noisy_x_train, 0., 1.)
    noisy_x_test = np.clip(noisy_x_test, 0., 1.)

    return noisy_x_train, x_train, noisy_x_test, x_test

# Model Architecture
def build_autoencoder():
    input_img = layers.Input(shape=(32, 32, 3))
    
    # Encoder
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = layers.Conv2DTranspose(16, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = models.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    
    return autoencoder

# Training
def train_autoencoder(autoencoder, noisy_x_train, x_train, noisy_x_test, x_test):
    history = autoencoder.fit(noisy_x_train, x_train,
                              epochs=50,
                              batch_size=64,
                              validation_data=(noisy_x_test, x_test))
    
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Evaluation
def evaluate_model(autoencoder, noisy_x_test, x_test):
    denoised_images = autoencoder.predict(noisy_x_test)
    
    for i in range(10):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title("Noisy Input")
        plt.imshow(noisy_x_test[i])
        plt.subplot(1, 3, 2)
        plt.title("Denoised Output")
        plt.imshow(denoised_images[i])
        plt.subplot(1, 3, 3)
        plt.title("Ground Truth")
        plt.imshow(x_test[i])
        plt.show()

        print(f"PSNR: {psnr(x_test[i], denoised_images[i]):.2f}")
        print(f"SSIM: {ssim(x_test[i], denoised_images[i], multichannel=True):.2f}")

# Deployment Function
def denoise_image(noisy_input):
    return autoencoder.predict(np.expand_dims(noisy_input, axis=0))[0]

# Main Execution
noisy_x_train, x_train, noisy_x_test, x_test = load_and_prepare_data()
autoencoder = build_autoencoder()
train_autoencoder(autoencoder, noisy_x_train, x_train, noisy_x_test, x_test)
evaluate_model(autoencoder, noisy_x_test, x_test)
