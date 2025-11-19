"""
Skin Lesion Classification App
8-Class Dermoscopic Image Classifier using ISIC2019-trained EfficientNet-B4
"""

import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
import requests
from io import BytesIO
import numpy as np
import plotly.graph_objects as go
import pandas as pd # <-- NEW: Used for model performance table

# -------------------------
# Configuration
# -------------------------
MODEL_URL = "https://huggingface.co/Skindoc/streamlit5/resolve/main/best_model_20251116_151842.pth"
MODEL_NAME = "tf_efficientnet_b4"
NUM_CLASSES = 8
IMG_SIZE = 384

CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'scc', 'vasc']

CLASS_INFO = {
    'akiec': {
        'full_name': 'Actinic Keratoses',
        'description': 'Pre-cancerous lesions caused by sun damage. Requires monitoring and treatment.',
        'risk': 'Medium',
        'color': '#FFA500'  # Orange
    },
    'bcc': {
        'full_name': 'Basal Cell Carcinoma',
        'description': 'Most common skin cancer. Slow-growing, rarely spreads, highly treatable.',
        'risk': 'High',
        'color': '#FF4444'  # Bright Red
    },
    'bkl': {
        'full_name': 'Benign Keratosis',
        'description': 'Non-cancerous skin growth. Generally harmless but may be removed for cosmetic reasons.',
        'risk': 'Low',
        'color': '#90EE90'  # Light Green
    },
    'df': {
        'full_name': 'Dermatofibroma',
        'description': 'Benign fibrous nodule. Usually harmless and does not require treatment.',
        'risk': 'Low',
        'color': '#87CEEB'  # Sky Blue
    },
    'mel': {
        'full_name': 'Melanoma',
        'description': 'Most dangerous skin cancer. Can spread rapidly. Requires immediate medical attention.',
        'risk': 'Critical',
        'color': '#8B0000'  # Dark Red/Maroon
    },
    'nv': {
        'full_name': 'Melanocytic Nevus',
        'description': 'Common moles. Generally benign but should be monitored for changes.',
        'risk': 'Low',
        'color': '#98FB98'  # Pale Green
    },
    'scc': {
        'full_name': 'Squamous Cell Carcinoma',
        'description': 'Second most common skin cancer. Can spread if untreated. Requires treatment.',
        'risk': 'High',
        'color': '#FF6347'  # Tomato Red
    },
    'vasc': {
        'full_name': 'Vascular Lesions',
        'description': 'Blood vessel abnormalities. Usually benign (e.g., cherry angiomas, hemangiomas).',
        'risk': 'Low',
        'color': '#DDA0DD'  # Plum
    }
}

# -------------------------
# Custom CSS for Professional Look
# -------------------------
def set_theme(background_color='#0E1117'):
    """Sets a consistent dark-themed style for a professional look."""
    css = f"""
    <style>
    /* 1. Global Background Color and Clean Up */
    .stApp {{
        background-color: {background_color};
        background-image: none;
    }}
    
    /* 2. Main Content Container - **Reduced Opacity for a darker, sleeker look** */
    .main .block-container {{
        background-color: rgba(18, 18, 18, 0.9); /* Slightly more opaque */
        padding-top: 3rem;  /* Reduced padding for less wasted space */
        padding-right: 3rem;
        padding-left: 3rem;
        padding-bottom: 3rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5); /* Added subtle shadow */
    }}
    
    /* 3. Text and Header Colors */
    h1, h2, h3, h4, .stMarkdown, .stText, label, p, .css-1456l0p, .css-1dp5vir {{
        color: #F0F2F6 !important;
    }}
    
    /* 4. Sidebar Contrast */
    [data-testid="stSidebar"] {{
        background-color: rgba(30, 30, 30, 0.95);
        color: #F0F2F6;
    }}

    /* 5. Streamlit Tabs Styling */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {{
        border-bottom: 1px solid #333;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
    }}
    
    /* 6. Custom Horizontal Rule */
    hr {{
        border-top: 1px solid #333;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# -------------------------
# Model Loading (Unchanged for brevity)
# -------------------------
@st.cache_resource
def load_model():
    """Load the trained model from HuggingFace"""
    try:
        # Download model weights
        # ... [existing model loading logic] ...
        with st.spinner("Downloading model (this may take a minute on first run)..."):
            response = requests.get(MODEL_URL)
            response.raise_for_status()

        # Load checkpoint
        checkpoint = torch.load(BytesIO(response.content), map_location='cpu')

        # Build model
        model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)

        model.eval()

        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# -------------------------
# Image Preprocessing & Prediction (Unchanged for brevity)
# -------------------------
def get_transform():
    # ... [existing transform logic] ...
    return transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.05)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def preprocess_image(image: Image.Image) -> torch.Tensor:
    # ... [existing preprocess logic] ...
    if image.mode != 'RGB':
        image = image.convert('RGB')

    transform = get_transform()
    tensor = transform(image).unsqueeze(0)
    return tensor

def predict_with_tta(model: torch.nn.Module, image_tensor: torch.Tensor, use_tta: bool = True) -> np.ndarray:
    # ... [existing prediction logic] ...
    with torch.no_grad():
        if use_tta:
            probs_list = [
                F.softmax(model(image_tensor), dim=1),
                F.softmax(model(torch.flip(image_tensor, dims=[3])), dim=1),
                F.softmax(model(torch.flip(image_tensor, dims=[2])), dim=1)
            ]
            probs = torch.stack(probs_list).mean(0)
        else:
            outputs = model(image_tensor)
            probs = F.softmax(outputs, dim=1)

    return probs.cpu().numpy()[0]


# -------------------------
# Visualization Utilities (Minor adjustments for aesthetics)
# -------------------------
def create_probability_chart(probabilities: np.ndarray, class_names: list) -> go.Figure:
    """Create an interactive bar chart of probabilities."""
    prob_class_pairs = list(zip(probabilities, class_names))
    prob_class_pairs.sort(key=lambda x: x[0], reverse=True)

    sorted_probs = [pair[0] for pair in prob_class_pairs]
    sorted_names = [pair[1] for pair in prob_class_pairs]

    sorted_full_names = [CLASS_INFO[name]['full_name'] for name in sorted_names]
    sorted_colors = [CLASS_INFO[name]['color'] for name in sorted_names]

    fig = go.Figure(data=[
        go.Bar(
            x=[p * 100 for p in sorted_probs],
            y=sorted_full_names,
            orientation='h',
            marker=dict(color=sorted_colors),
            text=[f'{p*100:.1f}%' for p in sorted_probs],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Confidence: %{x:.1f}%<extra></extra>' # Professional tooltip
        )
    ])

    fig.update_layout(
        title=None, # Title is now outside the chart for consistency
        xaxis_title="Confidence (%)",
        yaxis_title=None,
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(30, 30, 30, 0.8)',
        paper_bgcolor='rgba(18, 18, 18, 0.1)',
        font=dict(color='#F0F2F6'),
        xaxis=dict(range=[0, 105])
    )

    return fig

def create_risk_indicator(top_class: str):
    """Create a risk level indicator HTML and return the risk level."""
    risk = CLASS_INFO[top_class]['risk']

    risk_colors = {
        'Low': '#4CAF50',
        'Medium': '#FFC107',
        'High': '#FF5722',
        'Critical': '#F44336'
    }

    color = risk_colors.get(risk, '#808080')

    # Modified HTML for a cleaner, inline look
    html = f"""
    <div style="display: flex; align-items: center; justify-content: space-between; padding: 10px 20px; border-radius: 8px; background-color: {color}; color: white; margin-bottom: 20px;">
        <span style="font-size: 1.2em; font-weight: bold; color: white !important;">Risk Level:</span>
        <span style="font-size: 1.5em; font-weight: bold; color: white !important;">{risk}</span>
    </div>
    """
    return html, risk

def get_performance_table():
    """Creates a professional DataFrame for model performance metrics."""
    data = {
        'Metric': ['Macro F1 Score', 'Macro AUC', 'Balanced Accuracy', 'Training Dataset'],
        'Value': ['0.845', '0.984', '0.836', 'ISIC2019 (25,331 images)']
    }
    df = pd.DataFrame(data).set_index('Metric')
    return df


# -------------------------
# Streamlit UI
# -------------------------
def main():
    # Page configuration (must be first)
    st.set_page_config(
        page_title="Skin Scanner AI Tool",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    set_theme()

    # --- HEADER (Cleaner look) ---
    st.markdown(
        """
        <div style="text-align: center;">
        <h1 style="color: #4ECDC4; margin-bottom: 0px;">🔬 Skin Scanner AI</h1>
        <p style='font-size: 16px; color: #aaa;'>
        **EfficientNet-B4** | 8-Class Dermoscopic Image Classifier
        </p>
        </div>
        <hr>
        """,
        unsafe_allow_html=True
    )

    # --- SIDEBAR (Refactored for better UX) ---
    with st.sidebar:
        st.header("⚙️ App Controls")
        
        # Settings moved to sidebar (global options)
        st.subheader("Prediction Settings")
        use_tta = st.checkbox("Use Test-Time Augmentation (TTA)", value=True,
                              help="A technique that averages predictions over slightly augmented versions of the input image, improving robustness and accuracy.")
        show_all_probabilities = st.checkbox("Show Detailed Probability Chart", value=True)

        st.divider()

        st.header("📊 Model Performance")
        # Display performance as a clean table (DataFrames are generally more professional than metrics)
        st.table(get_performance_table())
        st.caption("Metrics from ISIC2019 Validation Set")

        st.divider()

        st.warning("""
        ⚠️ **Medical Disclaimer**
        
        This tool is for **educational/research** use only. It is **NOT** a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified dermatologist.
        """)

    # Load model
    model = load_model()

    if model is None:
        st.error("Failed to load model. Please refresh the page.")
        return

    # Main content: Use tabs for organized presentation
    tab_upload, tab_info = st.tabs(["🚀 Classification Tool", "📚 Lesion Info"])

    # --- TAB 1: CLASSIFICATION TOOL ---
    with tab_upload:
        st.subheader("1. Upload Dermoscopic Image")
        
        uploaded_file = st.file_uploader(
            "Choose a high-quality dermoscopic image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image captured using a dermatoscope."
        )

        if uploaded_file is not None:
            try:
                # Display image and results using columns
                col_img, col_pred = st.columns([1, 1])

                with col_img:
                    st.subheader("2. Image Preview")
                    image = Image.open(uploaded_file)
                    st.image(image, use_column_width=True)
                    st.caption(f"Image size: {image.size[0]} x {image.size[1]} pixels | Model Input Size: {IMG_SIZE}x{IMG_SIZE}")

                with col_pred:
                    st.subheader("3. Analysis Result")

                    # Make prediction
                    with st.spinner("Analyzing image..."):
                        image_tensor = preprocess_image(image)
                        probabilities = predict_with_tta(model, image_tensor, use_tta=use_tta)

                    # Get top prediction
                    top_idx = np.argmax(probabilities)
                    top_class = CLASS_NAMES[top_idx]
                    top_prob = probabilities[top_idx]

                    # Display risk indicator
                    risk_html, risk_level = create_risk_indicator(top_class)
                    st.markdown(risk_html, unsafe_allow_html=True)

                    # Display top prediction
                    st.markdown(f"**Predicted Type:** <h2 style='color: {CLASS_INFO[top_class]['color']}'>{CLASS_INFO[top_class]['full_name']}</h2>", unsafe_allow_html=True)
                    st.markdown(f"**Confidence:** <span style='font-size: 1.5em;'>{top_prob*100:.1f}%</span>", unsafe_allow_html=True)
                    st.progress(float(top_prob))

                    # Clinical recommendations (moved below main prediction)
                    st.markdown("---")
                    st.subheader("🩺 Clinical Recommendations")
                    
                    if risk_level in ['Critical', 'High']:
                        st.error(f"**⚠️ URGENT: This lesion shows characteristics of {CLASS_INFO[top_class]['full_name']}**\n\n- Schedule a **dermatologist appointment immediately**.\n- Early detection is crucial.")
                    elif risk_level == 'Medium':
                        st.warning(f"**⚡ MEDIUM PRIORITY: This lesion shows characteristics of {CLASS_INFO[top_class]['full_name']}**\n\n- Schedule a dermatologist appointment within **1-2 weeks**.\n- Monitor for any changes.")
                    else:
                        st.info(f"**✓ LOW PRIORITY: This lesion appears to be {CLASS_INFO[top_class]['full_name']}**\n\n- Continue regular skin monitoring.\n- Annual dermatology check-ups recommended.")

                # Detailed results tab
                st.markdown("<hr>", unsafe_allow_html=True)
                
                tab_desc, tab_prob, tab_top3 = st.tabs(["📝 Description", "📊 Full Probabilities", "🥇 Top 3 Breakdown"])
                
                with tab_desc:
                    st.markdown(f"### {CLASS_INFO[top_class]['full_name']}")
                    st.markdown(CLASS_INFO[top_class]['description'])
                    st.markdown(f"**Associated Risk:** `{risk_level}`")

                with tab_prob:
                    if show_all_probabilities:
                        st.subheader("Detailed Probability Distribution Across All Classes")
                        fig = create_probability_chart(probabilities, CLASS_NAMES)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Detailed chart disabled. Enable it in the 'App Controls' sidebar.")

                with tab_top3:
                    st.subheader("Top 3 Model Predictions")
                    top_3_idx = np.argsort(probabilities)[::-1][:3]
                    cols = st.columns(3)
                    for i, idx in enumerate(top_3_idx):
                        class_name = CLASS_NAMES[idx]
                        prob = probabilities[idx]
                        
                        with cols[i]:
                            # Simplified HTML for clarity
                            st.markdown(f"""
                            <div style="padding: 15px; border-radius: 8px; border-left: 5px solid {CLASS_INFO[class_name]['color']}; background-color: rgba(30, 30, 30, 0.7);">
                                <h5 style="margin-top: 0; color: {CLASS_INFO[class_name]['color']} !important;">#{i+1}: {CLASS_INFO[class_name]['full_name']}</h5>
                                <p><strong>Confidence:</strong> {prob*100:.1f}%</p>
                                <p><strong>Risk:</strong> {CLASS_INFO[class_name]['risk']}</p>
                            </div>
                            """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"⚠️ An error occurred while processing the image. Error details: {str(e)}")
                st.info("Please ensure the image is a valid JPG/PNG file and try again.")
        else:
            st.info("""
            👆 **Please upload a dermoscopic image to begin analysis**
            
            **Tips for best results:** Use high-quality dermoscopic images. Not validated for subungal or mucousal lesions.
            """)

    # --- TAB 2: LESION INFO ---
    with tab_info:
        st.header("Lesion Category Information")
        st.markdown("A brief guide to the 8 classifications used by this model.")
        
        # Use st.expander for a clean, collapsible list
        for key, info in CLASS_INFO.items():
            full_name = info['full_name']
            color = info['color']
            risk = info['risk']
            
            with st.expander(f"**{full_name}** ({key.upper()}) - Risk: {risk}", expanded=(risk in ['Critical', 'High'])):
                st.markdown(f"**Clinical Description:** {info['description']}")
                st.markdown(f"**Associated Risk Level:** <span style='color: {color}; font-weight: bold;'>{risk}</span>", unsafe_allow_html=True)
                
        st.markdown("---")
        st.subheader("What is a dermoscopic image?")
        st.markdown("""
        Dermoscopic images are captured using a **dermatoscope**, a specialized tool that uses magnification and polarized light to examine skin patterns beneath the surface, enabling more accurate diagnoses.
        """)
        

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #999; padding: 10px;">
        <p><strong>Model:</strong> EfficientNet-B4 | Trained on 25,331 ISIC2019 images | 8-class classification</p>
        <p><strong>Developed by:</strong> Dr Tom Hutchinson, Oxford, England | For educational and research purposes only</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
