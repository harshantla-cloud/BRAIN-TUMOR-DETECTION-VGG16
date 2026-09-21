import os
import urllib.request
import numpy as np
from PIL import Image
import streamlit as st
from tensorflow.keras.models import load_model

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Brain Tumor Detection | VGG16",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main container */
    .main {
        padding-top: 1rem;
    }

    /* Header */
    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #666;
        margin-bottom: 30px;
    }

    /* Prediction card */
    .prediction-card {
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #ddd;
        margin-top: 15px;
    }

    /* Section titles */
    .section-title {
        font-size: 24px;
        font-weight: 600;
        margin-top: 20px;
    }

    /* Sidebar */
    .sidebar-title {
        font-size: 22px;
        font-weight: 700;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #777;
        font-size: 13px;
        margin-top: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_PATH = "brain_tumor_model.keras"
MODEL_URL = "https://github.com/harshantla-cloud/BRAIN-TUMOR-DETECTION-VGG16/releases/download/v1.0/brain_tumor_model.keras"

IMAGE_SIZE = (128, 128)

CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary Tumor"]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

  st.markdown(
      '<div class="sidebar-title">🧠 Model Information</div>',
      unsafe_allow_html=True,
  )

  st.write("")

  st.markdown("### Architecture")
  st.write("VGG16")

  st.markdown("### Task")
  st.write("Multi-class Brain MRI Classification")

  st.markdown("### Input Size")
  st.write("128 × 128 pixels")

  st.markdown("### Classes")
  for class_name in CLASS_NAMES:
    st.write(f"• {class_name}")

  st.divider()

  st.markdown("### Supported Images")
  st.write("JPG, JPEG, PNG")

  st.divider()

  st.info(
      "This application is developed for educational "
      "and demonstration purposes."
  )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🧠 Brain Tumor Detection</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">VGG16-based Brain MRI Image'
    " Classification</div>",
    unsafe_allow_html=True,
)


# ============================================================
# MODEL LOADING FUNCTION (WITH AUTO-DOWNLOAD)
# ============================================================


@st.cache_resource
def load_brain_model():
  # Agar local disk par model nahi hai, toh GitHub release se download karega
  if not os.path.exists(MODEL_PATH):
    with st.spinner(
        "Downloading model weights (122 MB)... please wait a moment."
    ):
      urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

  model = load_model(MODEL_PATH)
  return model


# ============================================================
# LOAD MODEL
# ============================================================

try:
  model = load_brain_model()
  st.success("✅ VGG16 model loaded successfully.")

except Exception as e:
  st.error("❌ Unable to load the trained model.")

  with st.expander("Show technical details"):
    st.exception(e)

  st.info(
      "Make sure that the model release URL is accessible or that"
      " 'brain_tumor_model.keras' is present."
  )

  st.stop()


# ============================================================
# INTRODUCTION
# ============================================================

st.markdown(
    """
    ### 📌 About the Application

    Upload a brain MRI image to obtain a prediction from the
    trained VGG16 deep learning model. The model classifies
    the image into one of four categories.
    """
)


# ============================================================
# IMAGE UPLOADER
# ============================================================

st.markdown(
    '<div class="section-title">📤 Upload MRI Image</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Select a brain MRI image",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear brain MRI image in JPG, JPEG, or PNG format.",
)


# ============================================================
# PREDICTION PIPELINE
# ============================================================

if uploaded_file is not None:

  try:

    # ----------------------------------------------------
    # LOAD IMAGE
    # ----------------------------------------------------

    image = Image.open(uploaded_file).convert("RGB")

    # ----------------------------------------------------
    # DISPLAY IMAGE
    # ----------------------------------------------------

    col1, col2 = st.columns([1, 1])

    with col1:

      st.markdown("### 🖼️ Uploaded MRI")

      st.image(
          image, caption="Uploaded Brain MRI", use_container_width=True
      )

    with col2:

      st.markdown("### 📋 Image Information")

      width, height = image.size

      st.write(f"**File name:** {uploaded_file.name}")
      st.write(f"**Original size:** {width} × {height}")
      st.write(f"**Image mode:** {image.mode}")

      st.write(
          "**Model input:** " f"{IMAGE_SIZE[0]} × {IMAGE_SIZE[1]}"
      )

    st.divider()

    # ----------------------------------------------------
    # PREPROCESS IMAGE
    # ----------------------------------------------------

    image_resized = image.resize(IMAGE_SIZE)

    image_array = np.asarray(image_resized, dtype=np.float32)

    # Normalize pixel values
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # ----------------------------------------------------
    # PREDICTION
    # ----------------------------------------------------

    with st.spinner("🔍 Analyzing MRI image..."):

      predictions = model.predict(image_array, verbose=0)

    # ----------------------------------------------------
    # VALIDATE MODEL OUTPUT
    # ----------------------------------------------------

    if predictions.shape[-1] != len(CLASS_NAMES):

      st.error(
          "Model output does not match the configured "
          "number of classes."
      )

      st.stop()

    # ----------------------------------------------------
    # GET PREDICTION
    # ----------------------------------------------------

    predicted_index = int(np.argmax(predictions[0]))

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = float(predictions[0][predicted_index]) * 100

    # ====================================================
    # RESULT
    # ====================================================

    st.markdown(
        '<div class="section-title">🔎 Prediction Result</div>',
        unsafe_allow_html=True,
    )

    result_col1, result_col2 = st.columns(2)

    with result_col1:

      st.success(f"### Prediction\n**{predicted_class}**")

    with result_col2:

      st.metric(
          label="Prediction Confidence", value=f"{confidence:.2f}%"
      )

    # ----------------------------------------------------
    # CONFIDENCE INTERPRETATION
    # ----------------------------------------------------

    if confidence >= 90:

      st.info(
          "The model produced a high-confidence prediction "
          "for this image."
      )

    elif confidence >= 70:

      st.info(
          "The model produced a moderate-to-high confidence "
          "prediction."
      )

    else:

      st.warning(
          "The model confidence is relatively low. "
          "This prediction should be treated cautiously."
      )

    st.divider()

    # ====================================================
    # CLASS PROBABILITIES
    # ====================================================

    st.markdown(
        '<div class="section-title">📊 Class Probabilities</div>',
        unsafe_allow_html=True,
    )

    probabilities = predictions[0]

    for index, class_name in enumerate(CLASS_NAMES):

      probability = float(probabilities[index])

      percentage = probability * 100

      st.write(f"**{class_name} — {percentage:.2f}%**")

      st.progress(min(max(probability, 0.0), 1.0))

    # ====================================================
    # MODEL SUMMARY
    # ====================================================

    st.divider()

    with st.expander("🔬 View Prediction Details"):

      st.write(f"**Predicted class index:** {predicted_index}")

      st.write(f"**Predicted class:** {predicted_class}")

      st.write(f"**Confidence:** {confidence:.2f}%")

      st.write(
          f"**Input dimensions:** {IMAGE_SIZE[0]} × {IMAGE_SIZE[1]} × 3"
      )

      st.write("**Normalization:** Pixel values divided by 255")

  except Exception as e:

    st.error("❌ An error occurred while processing the image.")

    with st.expander("Show technical details"):

      st.exception(e)


# ============================================================
# INSTRUCTIONS WHEN NO IMAGE IS UPLOADED
# ============================================================

else:

  st.info("👆 Upload an MRI image above to start the prediction.")

  st.markdown(
      """
        **Prediction categories:**

        - 🧠 Glioma
        - 🧠 Meningioma
        - ✅ No Tumor
        - 🧠 Pituitary Tumor
        """
  )


# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.warning(
    """
    ⚠️ **Medical Disclaimer**

    This application is an educational machine learning project.
    It is not a medical device and should not be used for
    diagnosis, treatment, or clinical decision-making.

    Always consult a qualified healthcare professional for
    medical interpretation of MRI scans.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Brain Tumor Detection using VGG16 • Deep Learning Project
    </div>
    """,
    unsafe_allow_html=True,
)