import tensorflow as tf
import tensorflow_datasets as tfds

# Step 1: Download the Food101 dataset
dataset_name = 'food101'
(train_ds, validation_ds), info = tfds.load(
    name=dataset_name,
    split=['train', 'validation'],
    with_info=True,
    as_supervised=True
)

# Step 2: Preprocess the dataset
IMG_SIZE = (128, 128)  # Adjust the size as needed
IMG_SHAPE = (128, 128, 3)

# Function to preprocess an image with efficient augmentation
def preprocess_image(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0  # Normalize pixel values to be between 0 and 1
    return image, label

# Apply preprocessing with augmentation to the datasets
train_ds = train_ds.map(preprocess_image)
validation_ds = validation_ds.map(preprocess_image)

# Shuffle and batch the datasets
BATCH_SIZE = 32
train_ds = train_ds.shuffle(buffer_size=10000).batch(BATCH_SIZE)
validation_ds = validation_ds.batch(BATCH_SIZE)