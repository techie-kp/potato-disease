# import streamlit as st
# import tensorflow as tf
# import numpy as np
# from io import BytesIO
# from PIL import Image

# def read_file_as_image(data) -> np.ndarray:
#     image = np.array(Image.open(BytesIO(data)))
#     return image


# #Tensorflow Model Prediction
# def model_prediction(test_image):
#     # MODEL = tf.keras.models.load_model("C:/Users/kap23/OneDrive/Desktop/potato-disease/saved_models/model.keras")
#     MODEL = tf.keras.models.load_model("saved_models/model.keras")
    
#     image = tf.keras.preprocessing.image.load_img(test_image,target_size=(256,256))
#     input_arr = tf.keras.preprocessing.image.img_to_array(image)
#     input_arr = np.array([input_arr]) #convert single image to batch
    
#     predictions = MODEL.predict(input_arr)
#     # predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
#     return np.argmax(predictions[0])

# #Sidebar
# st.sidebar.title("Dashboard")
# app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

# #Main Page
# if(app_mode=="Home"):
#     st.header("PLANT DISEASE RECOGNITION SYSTEM")
#     image_path = "home_page.jpeg"
#     st.image(image_path,use_column_width=True)
#     st.markdown("""
#     Welcome to the Plant Disease Recognition System! 🌿🔍
    
#     Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

#     ### How It Works
#     1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
#     2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
#     3. **Results:** View the results and recommendations for further action.

#     ### Why Choose Us?
#     - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
#     - **User-Friendly:** Simple and intuitive interface for seamless user experience.
#     - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

#     ### Get Started
#     Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

#     ### About Us
#     Learn more about the project, our team, and our goals on the **About** page.
#     """)

# #About Project
# elif(app_mode=="About"):
#     st.header("About")
#     st.markdown("""
#                 #### About Dataset
#                 This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
#                 This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
#                 A new directory containing 33 test images is created later for prediction purpose.
#                 #### Content
#                 1. train (70295 images)
#                 2. test (33 images)
#                 3. validation (17572 images)

#                 """)

# #Prediction Page
# elif(app_mode=="Disease Recognition"):
#     st.header("Disease Recognition")
#     test_image = st.file_uploader("Choose an Image:")
#     if(st.button("Show Image")):
#         st.image(test_image,width=4,use_column_width=True)
#     #Predict button
#     if(st.button("Predict")):
#         st.snow()
#         st.write("Our Prediction")
#         result_index = model_prediction(test_image)
#         #Reading Labels
#         CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
        
#         st.success("Model is Predicting it's a {}".format(CLASS_NAMES[result_index]))


# import streamlit as st
# import tensorflow as tf
# import numpy as np
# from io import BytesIO
# from PIL import Image

# def read_file_as_image(data) -> np.ndarray:
#     image = np.array(Image.open(BytesIO(data)))
#     return image

# # Tensorflow Model Prediction
# def model_prediction(test_image):
#     MODEL = tf.keras.models.load_model("saved_models/model.keras")
    
#     image = tf.keras.preprocessing.image.load_img(test_image, target_size=(256, 256))
#     input_arr = tf.keras.preprocessing.image.img_to_array(image)
#     input_arr = np.array([input_arr])  # Convert single image to batch
    
#     predictions = MODEL.predict(input_arr)
#     return np.argmax(predictions[0])

# # Sidebar
# st.sidebar.title("Plant Disease Recognition")
# st.sidebar.image("sidelogo.jpg", use_column_width=True)
# app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# # Main Page
# if app_mode == "Home":
#     st.title("🌿 Plant Disease Recognition System 🌿")
#     st.image("home_page.jpeg", use_column_width=True)
#     st.markdown("""
#     Welcome to the **Plant Disease Recognition System**! Our mission is to help identify plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

#     ### How It Works
#     1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
#     2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
#     3. **Results:** View the results and recommendations for further action.

#     ### Why Choose Us?
#     - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
#     - **User-Friendly:** Simple and intuitive interface for seamless user experience.
#     - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

#     ### Get Started
#     Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

#     ### About Us
#     Learn more about the project, our team, and our goals on the **About** page.
#     """)
# elif app_mode == "About":
#     st.title("About the Project")
#     st.markdown("""
#     #### About Dataset
#     This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this GitHub repo.
    
#     This dataset consists of about 87K RGB images of healthy and diseased crop leaves, categorized into 38 different classes. The total dataset is divided into an 80/20 ratio of training and validation sets, preserving the directory structure. A new directory containing 33 test images is created later for prediction purposes.

#     #### Content
#     - **Train:** 70,295 images
#     - **Test:** 330 images
#     - **Validation:** 17,572 images
#     """)
# elif app_mode == "Disease Recognition":
#     st.title("Disease Recognition")
#     st.markdown("Upload an image of a plant to detect if it has any disease.")
#     test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
    
#     if test_image is not None:
#         st.image(test_image, use_column_width=True)
#         if st.button("Predict"):
#             st.snow()
#             st.markdown("### Our Prediction")
#             result_index = model_prediction(test_image)
#             CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
#             st.success(f"The model predicts: **{CLASS_NAMES[result_index]}**")

# # Footer
# st.sidebar.markdown("""
# ---
# Developed by **Khush Patel** [GitHub](https://github.com/techie-kp) | [LinkedIn](https://www.linkedin.com/in/khush-patel-1b2193291)
# """)


import streamlit as st
import tensorflow as tf
import numpy as np
from io import BytesIO
from PIL import Image

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

# Tensorflow Model Prediction
def model_prediction(test_image):
    MODEL = tf.keras.models.load_model("saved_models/model.keras")
    
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(256, 256))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    
    predictions = MODEL.predict(input_arr)
    return np.argmax(predictions[0])

# Remedies and nurturing tips
def get_suggestion(disease_name):
    suggestions = {
        "Early Blight": "Early Blight remedy: Remove affected leaves, use fungicides, and ensure proper spacing between plants to improve air circulation.",
        "Late Blight": "Late Blight remedy: Remove and destroy infected plants, avoid overhead watering, and use fungicides.",
        "Healthy": "Healthy plant nurturing tips: Regular watering, proper sunlight, and balanced fertilization will keep your plant healthy."
    }
    return suggestions.get(disease_name, "No suggestion available.")

# Sidebar
st.sidebar.title("Plant Disease Recognition")
st.sidebar.image("sidelogo.jpg", use_column_width=True)
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.title("🌿 Plant Disease Recognition System 🌿")
    st.image("home_page.jpeg", use_column_width=True)
    st.markdown("""
    Welcome to the **Plant Disease Recognition System**! Our mission is to help identify plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)
elif app_mode == "About":
    st.title("About the Project")
    st.markdown("""
    #### About Dataset
    This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this GitHub repo.
    
    This dataset consists of about 87K RGB images of healthy and diseased crop leaves, categorized into 38 different classes. The total dataset is divided into an 80/20 ratio of training and validation sets, preserving the directory structure. A new directory containing 33 test images is created later for prediction purposes.

    #### Content
    - **Train:** 70,295 images
    - **Test:** 33 images
    - **Validation:** 17,572 images
    """)
elif app_mode == "Disease Recognition":
    st.title("Disease Recognition")
    st.markdown("Upload an image of a plant to detect if it has any disease.")
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
    
    if test_image is not None:
        st.image(test_image, use_column_width=True)
        if st.button("Predict"):
            st.snow()
            st.markdown("### Our Prediction")
            result_index = model_prediction(test_image)
            CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
            prediction = CLASS_NAMES[result_index]
            st.success(f"The model predicts: **{prediction}**")
            suggestion = get_suggestion(prediction)
            st.info(suggestion)

# Footer
st.sidebar.markdown("""
---
Developed by **Khush Patel**. [GitHub](https://github.com/techie-kp) | [LinkedIn](https://www.linkedin.com/in/khush-patel-1b2193291)
""")

