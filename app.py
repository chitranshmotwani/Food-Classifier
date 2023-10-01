import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('saved_models/food_classifier_model_retrained_again.h5')
class_names = ['apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare',
                'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito',
                'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake',
                'ceviche', 'cheesecake', 'cheese_plate', 'chicken_curry', 'chicken_quesadilla',
                'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder',
                'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes',
                'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots',
                'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras', 'french_fries',
                'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice', 'frozen_yogurt',
                'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich', 'grilled_salmon',
                'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup', 'hot_dog', 'huevos_rancheros',
                'hummus', 'ice_cream', 'lasagna', 'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese',
                'macarons', 'miso_soup', 'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters',
                'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck', 'pho', 'pizza',
                'pork_chop', 'poutine', 'prime_rib', 'pulled_pork_sandwich', 'ramen', 'ravioli',
                'red_velvet_cake', 'risotto', 'samosa', 'sashimi', 'scallops', 'seaweed_salad',
                'shrimp_and_grits', 'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls',
                'steak', 'strawberry_shortcake', 'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare',
                'waffles', 'wedding_cake', 'wontons', 'yakitori']

# Define functions for model inference
def preprocess_image(image):
    img = Image.open(image).convert('RGB')
    img = img.resize((128, 128))  # Adjust dimensions as needed for your model
    img = np.asarray(img) / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img
def predict(image):
    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = prediction[0][np.argmax(prediction)]
    return predicted_class, confidence

# Streamlit UI
st.title('Food Image Classifier')
st.write('Upload a food image and get a prediction!')

uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write('')
    st.write('Classifying...')

    # Make predictions
    predicted_class, confidence = predict(uploaded_file)
    st.write('Prediction:', predicted_class)
    st.write('Confidence:', round(confidence * 100, 2), '%')
