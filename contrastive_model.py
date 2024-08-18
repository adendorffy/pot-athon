import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import pandas as pd
import os
import random
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Directory where images are stored
image_dir = 'train_images'
image_files = os.listdir(image_dir)  # List of all image filenames
random.shuffle(image_files)  # Shuffle to randomize pairs

img_height = 598
img_width = 525

# Placeholder lists for image pairs and labels
image_pairs = []
labels = []

# Function to determine if two images are similar (this is just an example)
def are_similar(img1, img2):
    # Implement your logic here to determine if two images are similar.
    return os.path.basename(img1).split('_')[0] == os.path.basename(img2).split('_')[0]

# Create pairs
for i in range(len(image_files)):
    for j in range(i + 1, len(image_files)):
        img1 = image_files[i]
        img2 = image_files[j]
        
        img1_path = os.path.join(image_dir, img1)
        img2_path = os.path.join(image_dir, img2)

        # Load and preprocess images
        img1_array = img_to_array(load_img(img1_path, target_size=(img_height, img_width))) / 255.0
        img2_array = img_to_array(load_img(img2_path, target_size=(img_height, img_width))) / 255.0

        # Append the pair to image_pairs
        image_pairs.append((img1_array, img2_array))
        
        # Append the label (1 for similar, 0 for dissimilar)
        label = 1 if are_similar(img1_path, img2_path) else 0
        labels.append(label)

# Convert lists to numpy arrays
image_pairs = np.array(image_pairs)
labels = np.array(labels)

print(f"Total pairs: {len(image_pairs)}")
print(f"Total labels: {len(labels)}")


def create_embedding_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(128, activation='relu')(x)
    return Model(inputs, outputs)

def create_siamese_network(input_shape):
    embedding_model = create_embedding_model(input_shape)
    
    input_a = layers.Input(shape=input_shape)
    input_b = layers.Input(shape=input_shape)
    
    # Generate embeddings
    embedding_a = embedding_model(input_a)
    embedding_b = embedding_model(input_b)
    
    # Compute the L2 distance between embeddings
    distance = layers.Lambda(lambda embeddings: tf.sqrt(tf.reduce_sum(tf.square(embeddings[0] - embeddings[1]), axis=1)))(
        [embedding_a, embedding_b]
    )
    
    # Output similarity score
    outputs = layers.Dense(1, activation='sigmoid')(distance)
    
    return Model([input_a, input_b], outputs)

def contrastive_loss(y_true, y_pred):
    margin = 1.0
    return tf.reduce_mean(y_true * tf.square(y_pred) + (1 - y_true) * tf.square(tf.maximum(margin - y_pred, 0)))

input_shape = (img_height, img_width, 3)
siamese_network = create_siamese_network(input_shape)
siamese_network.compile(optimizer='adam', loss=contrastive_loss, metrics=['accuracy'])

# Example: Create dummy pairs and labels
pairs = []
labels = []

# Convert image pairs to numpy arrays and scale them
image_pairs = [(
    np.array(tf.image.resize(tf.io.decode_image(tf.io.read_file(img1)), (img_height, img_width))) / 255.0,
    np.array(tf.image.resize(tf.io.decode_image(tf.io.read_file(img2)), (img_height, img_width))) / 255.0
) for img1, img2 in pairs]

labels = np.array(labels)

# Convert to TensorFlow datasets
dataset = tf.data.Dataset.from_tensor_slices(((image_pairs[0], image_pairs[1]), labels))
dataset = dataset.shuffle(buffer_size=len(labels)).batch(batch_size)

history = siamese_network.fit(dataset, epochs=10)

def predict_similarity(image1, image2):
    embedding_model = siamese_network.layers[2]  # Assuming the embedding model is the third layer
    embedding_a = embedding_model.predict(image1[np.newaxis, ...])
    embedding_b = embedding_model.predict(image2[np.newaxis, ...])
    distance = np.sqrt(np.sum(np.square(embedding_a - embedding_b)))
    return distance

# Example usage
similarity_score = predict_similarity(image1, image2)
print(f'Similarity score: {similarity_score}')
