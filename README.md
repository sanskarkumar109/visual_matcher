# 🖼️ Visual Product Matcher

A simple **Streamlit-based Visual Search App** that allows users to upload an image or paste a URL to find **visually similar products** from a local image database using **ResNet50** feature embeddings and **cosine similarity**.

---

## 🚀 Features

- 🔍 Upload image or use image URL for search  
- 🧠 Extract visual features using **ResNet50 (ImageNet pretrained)**  
- 🗂️ Automatically indexes all images in the `/db` folder  
- ⚡ Finds and displays top visually similar products  
- 🎚️ Adjustable similarity threshold and number of results  
- 💾 Feature caching for fast reloads (`.npy` files)  
- 🖥️ Clean Streamlit interface

---

## 📁 Folder Structure

visual_search/

├── main.py # Main Streamlit application

├── db/ # Product image database (your images go here)

│ ├── image1.jpg

│ ├── image2.png

│ └── ...

├── db_features.npy # Cached feature vectors (auto-created)

├── db_image_paths.npy # Cached image paths (auto-created)

├── requirements.txt # Python dependencies

└── README.md # Documentation


## 🧩 Installation

1. **Clone the repository**
   git clone https://github.com/sanskarkumar109/visual_matcher.git
   cd visual_product_matcher
Create a virtual environment

python -m venv venv
source venv/bin/activate    # For Linux/Mac
venv\Scripts\activate       # For Windows

Install dependencies

pip install -r requirements.txt
Add product images
Place at least 10–50 product images (JPEG, PNG) into the db/ folder.

▶️ Run the App
streamlit run main.py
Then open the provided local URL (e.g., http://localhost:8501) in your browser.

🧠 How It Works
Feature Extraction

Uses ResNet50 pretrained on ImageNet.

Converts each image into a 2048-dimensional embedding vector.

Normalizes vectors for cosine similarity comparison.

Database Indexing

Automatically scans the db/ directory for images.

Extracts and caches features into .npy files for reuse.

Similarity Matching

The query image’s embedding is compared against all cached embeddings.

Results are sorted by cosine similarity.

Top N or threshold-filtered results are displayed with similarity scores.

🧰 Dependencies
Add this to your requirements.txt file:

streamlit
tensorflow
scikit-learn
numpy
pillow
requests

💡 Example Use Cases
Visual product search for e-commerce

Duplicate image finder

Fashion or art similarity lookup

Image-based catalog browsing
