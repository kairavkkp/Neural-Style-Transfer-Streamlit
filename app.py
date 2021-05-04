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

# For supressing warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

st.write("""
# Neural Style Transfer
""")


@st.cache
def load_model():
    hub_model = hub.load(
        'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    return hub_model


model_load_state = st.text('Loading Model...')
model = load_model()
# Notify the reader that the data was successfully loaded.
model_load_state.text('Loading Model...done!')


content_image, style_image = st.beta_columns(2)

with content_image:
    st.write('## Content Image...')
    chosen_content = st.radio(
        '  ',
        ("Upload", "URL"))
    if chosen_content == 'Upload':
        st.write(f"You choosed {chosen_content}!")
        content_image_file = st.file_uploader(
            "Pick a Content image", type=("png", "jpg"))
        try:
            content_image_file = content_image_file.read()
            content_image_file = transform_img(content_image_file)
        except:
            pass
    elif chosen_content == 'URL':
        st.write(f"You choosed {chosen_content}!")
        url = st.text_input('URL for the content image.')
        try:
            content_path = tf.keras.utils.get_file(
                os.path.join(os.getcwd(), 'content.jpg'), url)
        except:
            pass
        try:
            content_image_file = load_img(content_path)

        except:
            pass
    try:
        st.write('Content Image...')
        st.image(imshow(content_image_file))
    except:
        pass

with style_image:
    st.write('## Style Image...')
    chosen_style = st.radio(
        ' ',
        ("Upload", "URL"))
    if chosen_style == 'Upload':
        st.write(f"You choosed {chosen_style}!")
        style_image_file = st.file_uploader(
            "Pick a Style image", type=("png", "jpg"))
        try:
            style_image_file = style_image_file.read()
            style_image_file = transform_img(style_image_file)
        except:
            pass
    elif chosen_style == 'URL':
        st.write(f"You choosed {chosen_style}!")
        url = st.text_input('URL for the style image.')
        try:
            style_path = tf.keras.utils.get_file(
                os.path.join(os.getcwd(), 'style.jpg'), url)
        except:
            pass
        try:
            style_image_file = load_img(style_path)

        except:
            pass
    try:
        st.write('Style Image...')
        st.image(imshow(style_image_file))
    except:
        pass

predict = st.button('Start Neural Style Transfer...')


if predict:
    try:
        stylized_image = model(tf.constant(
            content_image_file), tf.constant(style_image_file))[0]

        final_image = tensor_to_image(stylized_image)
    except:
        stylized_image = model(tf.constant(
            tf.convert_to_tensor(content_image_file[:, :, :, :3])
        ),
            tf.constant(
                tf.convert_to_tensor(style_image_file[:, :, :, :3])
        )
        )[0]

        final_image = tensor_to_image(stylized_image)

    st.write('Resultant Image...')
    st.image(final_image)

    try:
        # Delete style.jpg and content.jpg
        os.remove("style.jpg")
        os.remove("content.jpg")
    except:
        pass

st.write('Made by Kairav with \u2764\ufe0f.')
st.write('Happy Coding.')
