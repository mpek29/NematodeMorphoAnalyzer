from PIL import Image
import numpy as np

# return en numer between 0 and 1, 0 being no waste
def get_is_there_waste(image, load_m_waste):
    # search if there is waste
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    #img_array = img_array / 255.0 
    return  1-load_m_waste.predict(img_array, verbose=0)[0][0]
