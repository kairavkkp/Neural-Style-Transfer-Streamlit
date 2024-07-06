import streamlit as st
import os
import tensorflow_hub as hub
from utils import load_img, transform_img, tensor_to_image, imshow
import tensorflow as tf
import numpy as np
from PIL import Image

# Only use the below code if you have low resources.
os.environ['CUDA_VISIBLE_DEVICES'] = ""
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

# For suppressing warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

st.write("""
# Neural Style Transfer
""")

# Singleton pattern for model loading
class ModelLoader:
    _model = None

    @staticmethod
    def get_model():
        if ModelLoader._model is None:
            ModelLoader._model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
        return ModelLoader._model

model_load_state = st.text('Loading Model...')
model = ModelLoader.get_model()
# Notify the reader that the data was successfully loaded.
model_load_state.text('Loading Model...done!')

def ensure_rgb(image):
    if image.shape[-1] == 4:
        # Convert RGBA to RGB
        image = image[..., :3]
    return image

content_image_col, style_image_col = st.columns(2)

with content_image_col:
    st.write('## Content Image...')
    chosen_content = st.radio('Choose content image source:', ("Upload", "URL"))
    if chosen_content == 'Upload':
        content_image_file = st.file_uploader("Pick a Content image", type=("png", "jpg"))
        if content_image_file:
            content_image_file = transform_img(content_image_file.read())
            content_image_file = ensure_rgb(content_image_file)
    elif chosen_content == 'URL':
        url = st.text_input('URL for the content image.')
        if url:
            try:
                content_path = tf.keras.utils.get_file('content.jpg', url)
                content_image_file = load_img(content_path)
                content_image_file = ensure_rgb(content_image_file)
            except Exception as e:
                st.error(f"Error loading image: {e}")

    if 'content_image_file' in locals():
        if content_image_file is not None:
            st.write('Content Image...')
            st.image(imshow(content_image_file))

with style_image_col:
    st.write('## Style Image...')
    chosen_style = st.radio('Choose style image source:', ("Upload", "URL"))
    if chosen_style == 'Upload':
        style_image_file = st.file_uploader("Pick a Style image", type=("png", "jpg"))
        if style_image_file:
            style_image_file = transform_img(style_image_file.read())
            style_image_file = ensure_rgb(style_image_file)
    elif chosen_style == 'URL':
        url = st.text_input('URL for the style image.')
        if url:
            try:
                style_path = tf.keras.utils.get_file('style.jpg', url)
                style_image_file = load_img(style_path)
                style_image_file = ensure_rgb(style_image_file)
            except Exception as e:
                st.error(f"Error loading image: {e}")

    if 'style_image_file' in locals():
        if style_image_file is not None:
            st.write('Style Image...')
            st.image(imshow(style_image_file))

predict = st.button('Start Neural Style Transfer...')

if predict:
    if 'content_image_file' in locals() and 'style_image_file' in locals():
        try:
            stylized_image = model(tf.constant(content_image_file), tf.constant(style_image_file))[0]
            final_image = tensor_to_image(stylized_image)
        except Exception as e:
            st.error(f"Error during style transfer: {e}")
        else:
            st.write('Resultant Image...')
            st.image(final_image)
        finally:
            # Delete style.jpg and content.jpg if they exist
            for img in ['style.jpg', 'content.jpg']:
                if os.path.exists(img):
                    os.remove(img)

st.write('Made by Kairav with \u2764\ufe0f.')
st.write('Happy Coding.')
