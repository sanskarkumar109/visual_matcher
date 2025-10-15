import os
import io
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from typing import Union
import requests
from urllib.parse import urlparse

# ----------------------------
# SUPPRESS TF LOGS
# ----------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ----------------------------
# CONFIGURATION
# ----------------------------
DB_DIR = "db"
FEATURES_FILE = "db_features.npy"
PATHS_FILE = "db_image_paths.npy"
VALID_EXT = (".jpg", ".jpeg", ".png")

st.set_page_config(page_title="Visual Product Matcher", layout="wide")

# ----------------------------
# MODEL LOADING
# ----------------------------
@st.cache_resource
def load_model() -> tf.keras.Model:
    return ResNet50(weights="imagenet", include_top=False, pooling="avg")

# ----------------------------
# IMAGE HELPERS
# ----------------------------
def is_url(path: str) -> bool:
    try:
        result = urlparse(path)
        return all([result.scheme in ("http", "https"), result.netloc])
    except:
        return False

def load_image_from_url(url: str) -> Union[Image.Image, None]:
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")
    except Exception as e:
        st.error(f"Failed to load image from URL: {e}")
        return None

def extract_features(img_source: Union[str, Image.Image, io.BytesIO], model: tf.keras.Model) -> Union[np.ndarray, None]:
    try:
        if isinstance(img_source, Image.Image):
            img = img_source
        elif isinstance(img_source, io.BytesIO):
            img = Image.open(img_source).convert("RGB")
        else:
            img = Image.open(img_source).convert("RGB")

        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = model.predict(img_array, verbose=0).flatten()
        features = features / np.linalg.norm(features)
        return features
    except Exception as e:
        st.error(f"Error extracting features from {img_source}: {e}")
        return None

# ----------------------------
# DATABASE BUILDING
# ----------------------------
def build_feature_database(model: tf.keras.Model):
    feature_list, image_paths = [], []

    for root, _, files in os.walk(DB_DIR):
        for fname in files:
            if fname.lower().endswith(VALID_EXT):
                path = os.path.join(root, fname)
                features = extract_features(path, model)
                if features is not None:
                    feature_list.append(features)
                    image_paths.append(path)

    if not feature_list:
        st.error(" No valid images found in the database!")
        return np.array([]), []

    feature_vectors = np.vstack(feature_list)
    np.save(FEATURES_FILE, feature_vectors)
    np.save(PATHS_FILE, np.array(image_paths))
    return feature_vectors, image_paths

# ----------------------------
# LOAD DATABASE
# ----------------------------
def load_feature_database(model: tf.keras.Model):
    current_images = []
    for root, _, files in os.walk(DB_DIR):
        for fname in files:
            if fname.lower().endswith(VALID_EXT):
                current_images.append(os.path.join(root, fname))

    rebuild = False
    if not os.path.exists(FEATURES_FILE) or not os.path.exists(PATHS_FILE):
        rebuild = True
    else:
        cached_paths = np.load(PATHS_FILE, allow_pickle=True).tolist()
        if set(current_images) != set(cached_paths):
            rebuild = True

    if rebuild:
        feature_vectors, image_paths = build_feature_database(model)
    else:
        feature_vectors = np.load(FEATURES_FILE)
        image_paths = np.load(PATHS_FILE, allow_pickle=True).tolist()

    return feature_vectors, image_paths

# ----------------------------
# FIND SIMILAR IMAGES
# ----------------------------
def find_similar_images(query_img: Image.Image,
                        feature_vectors: np.ndarray,
                        image_paths: list[str],
                        model: tf.keras.Model):
    query_feat = extract_features(query_img, model)
    if query_feat is None or len(feature_vectors) == 0:
        return []

    similarities = cosine_similarity([query_feat], feature_vectors)[0]
    results = [(path, score) for path, score in zip(image_paths, similarities)]
    results.sort(key=lambda x: x[1], reverse=True)
    return results

# ----------------------------
# DISPLAY RESULTS
# ----------------------------
def display_results(results, min_score: float, top_n: int, show_all: bool):
    filtered = [r for r in results if r[1] >= min_score]
    if not filtered:
        st.warning("No images match the similarity threshold.")
        return

    if not show_all:
        filtered = filtered[:top_n]

    cols = st.columns(3)
    for i, (path, score) in enumerate(filtered):
        with cols[i % 3]:
            st.image(path, caption=f"Score: {score:.2f}", use_column_width=True)

# ----------------------------
# STREAMLIT UI
# ----------------------------
def main():
    st.title("Visual Product Matcher")
    st.write("Upload an image, paste a direct image URL, or use a local path to find visually similar products.")

    # Load model
    with st.spinner("Loading pretrained model..."):
        model = load_model()

    # Load database
    if not os.path.exists(DB_DIR):
        st.error(f"Database folder '{DB_DIR}' not found! Add product images.")
        st.stop()

    with st.spinner("Preparing image database..."):
        feature_vectors, image_paths = load_feature_database(model)
    st.success(f"Loaded {len(image_paths)} images from database.")

    # User Inputs
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    url_input = st.text_input("Or paste a direct image URL or local path:")

    min_score = st.slider("Minimum Similarity Score", 0.0, 1.0, 0.4, 0.01)
    top_n = st.slider("Top N Results", 1, 20, 5)
    show_all = st.checkbox("Show all matching results", value=False)

    query_img = None
    if uploaded_file is not None:
        query_img = Image.open(uploaded_file).convert("RGB")
    elif url_input.strip() != "":
        url_input = url_input.strip()
        if is_url(url_input):
            query_img = load_image_from_url(url_input)
        else:
            try:
                query_img = Image.open(url_input).convert("RGB")
            except Exception as e:
                st.error(f"Failed to load image from local path: {e}")

    if query_img:
        st.image(query_img, caption="Query Image", use_column_width=True)
        with st.spinner("Finding similar images..."):
            results = find_similar_images(query_img, feature_vectors, image_paths, model)
        display_results(results, min_score, top_n, show_all)
    else:
        st.info(" Upload an image, paste a URL, or use a local path to start searching.")

if __name__ == "__main__":
    main()






