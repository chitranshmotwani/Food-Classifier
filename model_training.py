import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from data_preprocessing import train_ds, validation_ds, IMG_SIZE

# Load the pre-trained MobileNetV2 model
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SIZE + (3,),
                                               include_top=False,
                                               weights='imagenet')

# Freeze the pre-trained layers to retain the learned features
base_model.trainable = False

# Add additional layers for fine-tuning
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(101, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
NUM_EPOCHS = 20
model.fit(train_ds, epochs=NUM_EPOCHS, validation_data=validation_ds)

# Save the trained model
model.save('saved_models/food_classifier_model.h5')
