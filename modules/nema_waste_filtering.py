import numpy as np

# Function to get the filtered mask using the model
def get_nemaWaste_filtered(image, filter_nemaWaste_model):
    img = image.resize((128, 128))  # Resize image for the model
    img = np.array(img) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add an extra dimension for model input
    return filter_nemaWaste_model.predict(img, verbose=0)