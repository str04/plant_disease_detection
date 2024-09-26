# import streamlit as st
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np
# import os
# from PIL import Image
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# # Paths for your dataset
# train_dir = r'C:\sem5\videoimage\project\dataset\train'

# # Set up ImageDataGenerator
# train_datagen = ImageDataGenerator(rescale=1./255)

# # Load train data just to get the class indices
# train_generator = train_datagen.flow_from_directory(
#     train_dir,
#     target_size=(150, 150),
#     batch_size=32,
#     class_mode='categorical'
# )

# # Load the pre-trained model
# model = load_model('best_model.keras')

# # Dynamically retrieve class indices
# class_indices = train_generator.class_indices
# class_labels = {v: k for k, v in class_indices.items()}

# # Define a function to predict image
# def predict_image(img_path, model):
#     img = image.load_img(img_path, target_size=(150, 150))
#     img_array = image.img_to_array(img) / 255.0  # Normalize image
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

#     prediction = model.predict(img_array)
#     predicted_class = np.argmax(prediction, axis=1)[0]

#     predicted_label = class_labels[predicted_class]
#     return predicted_label

# # Streamlit app
# st.title("Plant Disease Detection")

# # Upload image
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Display the uploaded image
#     image_pil = Image.open(uploaded_file)
#     st.image(image_pil, caption="Uploaded Image", use_column_width=True)

#     # Save uploaded image temporarily
#     img_path = os.path.join("temp.jpg")
#     with open(img_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())
#     # Predict the image class
#     st.write("Classifying...")
#     predicted_disease = predict_image(img_path, model)
#     st.write(f"Predicted Disease: **{predicted_disease}**")

#     # Clean up the temporary file
#     os.remove(img_path)

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths for your dataset
train_dir = r'C:\sem5\videoimage\project\dataset\train'

# Set up ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255)

# Load train data just to get the class indices
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# Load the pre-trained model
model = load_model('best_model.keras')

# Dynamically retrieve class indices
class_indices = train_generator.class_indices
class_labels = {v: k for k, v in class_indices.items()}

# Updated function to predict the image with percentage confidence
def predict_image_with_percentage(img_path, model, class_indices):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Make the prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    confidence = np.max(prediction) * 100  # Get the highest confidence score (as percentage)
    
    # Map class index back to class name
    class_labels = {v: k for k, v in class_indices.items()}
    predicted_label = class_labels[predicted_class[0]]
    
    # Determine if the predicted class is healthy or diseased
    if "healthy" in predicted_label:
        return f"The leaf is {confidence:.2f}% healthy."
    else:
        return f"The leaf is affected by {predicted_label} and {confidence:.2f}% affected."

# Streamlit app
st.title("Plant Disease Detection")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image_pil = Image.open(uploaded_file)
    st.image(image_pil, caption="Uploaded Image", use_column_width=True)

    # Save uploaded image temporarily
    img_path = os.path.join("temp.jpg")
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Predict the image class with percentage
    st.write("Classifying...")
    prediction_result = predict_image_with_percentage(img_path, model, class_indices)
    st.write(f"Prediction Result: **{prediction_result}**")

    # Clean up the temporary file
    os.remove(img_path)

