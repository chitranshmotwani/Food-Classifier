import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from data_preprocessing import train_ds, validation_ds

# Load the saved model
model = load_model('saved_models/food_classifier_model_retrained.h5')

# Modify the model to include data augmentation and fine-tune more layers
base_model = model.layers[0]
base_model.trainable = True

# Compile the model with a lower learning rate
model.compile(optimizer=Adam(learning_rate=0.00001),  # Adjust the learning rate as needed
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Define callbacks for early stopping and learning rate adjustment
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-7)

# Train the model with callbacks
NUM_EPOCHS = 50  # Adjust the number of epochs as needed
model.fit(train_ds,
          epochs=NUM_EPOCHS,
          validation_data=validation_ds,
          callbacks=[early_stopping, reduce_lr])

# Save the retrained model
model.save('saved_models/food_classifier_model_retrained_again.h5')
