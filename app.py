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
        'color': '#FFA500'  # Orange
    },
    'bcc': {
        'full_name': 'Basal Cell Carcinoma',
        'description': 'Most common skin cancer. Slow-growing, rarely spreads, highly treatable.',
        'risk': 'High',
        'color': '#FF4444'  # Bright Red
    },
    'bkl': {
        'full_name': 'Benign Keratosis',
        'description': 'Non-cancerous skin growth. Generally harmless but may be removed for cosmetic reasons.',
        'risk': 'Low',
        'color': '#90EE90'  # Light Green
    },
    'df': {
        'full_name': 'Dermatofibroma',
        'description': 'Benign fibrous nodule. Usually harmless and does not require treatment.',
        'risk': 'Low',
        'color': '#87CEEB'  # Sky Blue
    },
    'mel': {
        'full_name': 'Melanoma',
        'description': 'Most dangerous skin cancer. Can spread rapidly. Requires immediate medical attention.',
        'risk': 'Critical',
        'color': '#8B0000'  # Dark Red/Maroon
    },
    'nv': {
        'full_name': 'Melanocytic Nevus',
        'description': 'Common moles. Generally benign but should be monitored for changes.',
        'risk': 'Low',
        'color': '#98FB98'  # Pale Green
    },
    'scc': {
        'full_name': 'Squamous Cell Carcinoma',
        'description': 'Second most common skin cancer. Can spread if untreated. Requires treatment.',
        'risk': 'High',
        'color': '#FF6347'  # Tomato Red
    },
    'vasc': {
        'full_name': 'Vascular Lesions',
        'description': 'Blood vessel abnormalities. Usually benign (e.g., cherry angiomas, hemangiomas).',
        'risk': 'Low',
        'color': '#DDA0DD'  # Plum
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
        background: linear-gradient(135deg, #0E1117 0%, #1a1a2e 100%);
        background-attachment: fixed;
    }}
    
    /* 2. Main Content Container - Professional styling */
    .main .block-container {{
        background-color: rgba(18, 18, 18, 0.95);
        padding-top: 2.5rem;
        padding-right: 3rem;
        padding-left: 3rem;
        padding-bottom: 2.5rem;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.05);
        max-width: 1400px;
    }}
    
    /* 3. Text and Header Colors */
    h1, h2, h3, h4, .stMarkdown, .stText, label, p {{
        color: #F0F2F6 !important;
    }}
    
    h1 {{
        font-weight: 700;
        letter-spacing: -0.5px;
    }}
    
    h2 {{
        font-weight: 600;
        margin-top: 1.5rem;
    }}
    
    /* 4. Sidebar Professional Styling */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, rgba(25, 25, 40, 0.98) 0%, rgba(30, 30, 45, 0.98) 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }}
    
    [data-testid="stSidebar"] .block-container {{
        padding-top: 1rem;
    }}
    
    /* 5. Logo Container Styling */
    .logo-container {{
        text-align: center;
        padding: 1.5rem 0 1rem 0;
        margin-bottom: 1rem;
        border-bottom: 2px solid rgba(78, 205, 196, 0.3);
    }}
    
    .logo-container img {{
        max-width: 180px;
        width: 100%;
        height: auto;
        filter: drop-shadow(0 4px 8px rgba(78, 205, 196, 0.3));
        transition: transform 0.3s ease;
    }}
    
    .logo-container img:hover {{
        transform: scale(1.02);
    }}
    
    /* 6. Improve button and input styling */
    .stButton > button {{
        background: linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(78, 205, 196, 0.3);
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(78, 205, 196, 0.4);
    }}
    
    /* 7. File uploader styling */
    [data-testid="stFileUploader"] {{
        border: 2px dashed rgba(78, 205, 196, 0.3);
        border-radius: 12px;
        padding: 1rem;
        background: rgba(78, 205, 196, 0.05);
        transition: all 0.3s ease;
    }}
    
    [data-testid="stFileUploader"]:hover {{
        border-color: rgba(78, 205, 196, 0.6);
        background: rgba(78, 205, 196, 0.1);
    }}
    
    /* 8. Metric cards styling */
    [data-testid="stMetricValue"] {{
        font-size: 2rem;
        font-weight: 700;
    }}
    
    /* 9. Tab styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 4px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }}
    
    /* 10. Progress bar styling */
    .stProgress > div > div > div {{
        background: linear-gradient(90deg, #4ECDC4 0%, #44A08D 100%);
    }}
    
    /* 11. Custom Horizontal Rule */
    hr {{
        border: none;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        margin: 2rem 0;
    }}
    
    /* 12. Info/Warning/Success boxes */
    .stAlert {{
        border-radius: 12px;
        border-left: 4px solid;
    }}
    
    /* 13. Table styling */
    .dataframe {{
        border-radius: 8px;
        overflow: hidden;
    }}
    
    /* 14. Remove default Streamlit header */
    header[data-testid="stHeader"] {{
        display: none;
    }}
    
    /* 15. Footer styling */
    footer {{
        display: none;
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
        # Download model weights with timeout
        response = requests.get(MODEL_URL, timeout=(10, 300))  # 10s connect, 300s read timeout
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
    except requests.exceptions.Timeout:
        # Return error info as tuple - can't use st.error() in cached function
        return ("timeout", "Model download timed out. Please check your internet connection and try again.")
    except requests.exceptions.RequestException as e:
        return ("download_error", f"Error downloading model: {str(e)}")
    except Exception as e:
        return ("error", f"Error loading model: {str(e)}")

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

    # --- HEADER (Professional look) ---
    st.markdown(
        """
        <div style="text-align: center; padding: 2rem 0 1.5rem 0;">
        <h1 style="color: #4ECDC4; margin-bottom: 0.5rem; font-size: 2.5rem; font-weight: 700;">
            🔬 Skin Scanner AI
        </h1>
        <p style='font-size: 1.1rem; color: #aaa; margin-bottom: 0.5rem;'>
            <strong>EfficientNet-B4</strong> | 8-Class Dermoscopic Image Classifier
        </p>
        <p style='font-size: 0.95rem; color: #777;'>
            Advanced AI-powered skin lesion analysis for dermatological assessment
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<hr style='margin: 0.5rem 0 2rem 0;'>", unsafe_allow_html=True)

    # Initialize session state for model if not exists
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'model' not in st.session_state:
        st.session_state.model = None

    # --- SIDEBAR (Professional layout with logo) ---
    with st.sidebar:
        # Logo at the top of sidebar
        logo_loaded = False
        for logo_filename in ["logo.png", "logo.jpg", "logo.jpeg"]:
            try:
                logo = Image.open(logo_filename)
                st.markdown('<div class="logo-container">', unsafe_allow_html=True)
                st.image(logo, use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                logo_loaded = True
                break
            except (FileNotFoundError, IOError, OSError):
                continue
        
        if not logo_loaded:
            # If no logo found, show a placeholder message
            st.markdown(
                """
                <div style="text-align: center; padding: 1rem 0; border-bottom: 2px solid rgba(78, 205, 196, 0.3);">
                    <p style="color: #4ECDC4; font-weight: 600; font-size: 1.2rem;">Skin Scanner AI</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.header("⚙️ App Controls")
        
        # Settings moved to sidebar (global options)
        st.subheader("🔧 Prediction Settings")
        use_tta = st.checkbox(
            "Use Test-Time Augmentation (TTA)", 
            value=True,
            help="Averages predictions over slightly augmented versions of the input image, improving robustness and accuracy."
        )
        show_all_probabilities = st.checkbox(
            "Show Detailed Probability Chart", 
            value=True,
            help="Display comprehensive probability distribution across all lesion classes."
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---", unsafe_allow_html=True)

        st.subheader("📊 Model Performance")
        # Display performance as a clean table
        st.dataframe(
            get_performance_table(),
            use_container_width=True,
            hide_index=False,
            height=160
        )
        st.caption("📈 Metrics from ISIC2019 Validation Set")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---", unsafe_allow_html=True)

        st.markdown(
            """
            <div style="background: rgba(255, 193, 7, 0.1); border-left: 4px solid #FFC107; 
                        padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                <h4 style="color: #FFC107; margin-top: 0;">⚠️ Medical Disclaimer</h4>
                <p style="color: #F0F2F6; font-size: 0.9rem; margin-bottom: 0;">
                    This tool is for <strong>educational/research</strong> use only. It is 
                    <strong>NOT</strong> a substitute for professional medical advice, diagnosis, 
                    or treatment. Always consult a qualified dermatologist.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Load model (cached, so only downloads once)
    if not st.session_state.model_loaded:
        with st.spinner("🔄 Loading AI model (this may take a minute on first run)..."):
            result = load_model()
            # Handle error tuples from load_model
            if isinstance(result, tuple):
                error_type, error_msg = result
                st.error(f"❌ {error_msg}")
                if error_type == "timeout":
                    st.info("💡 The model download timed out. Please check your internet connection and try refreshing the page.")
                else:
                    st.info("💡 Please try refreshing the page. If the problem persists, check your internet connection.")
                st.stop()
                model = None
            else:
                model = result
            st.session_state.model = model
            st.session_state.model_loaded = True
    else:
        model = st.session_state.model

    if model is None:
        st.error("❌ Failed to load model. Please refresh the page.")
        st.stop()

    # Main content: Use tabs for organized presentation
    tab_upload, tab_info = st.tabs(["🚀 Classification Tool", "📚 Lesion Info"])

    # --- TAB 1: CLASSIFICATION TOOL ---
    with tab_upload:
        st.markdown("### 📤 Step 1: Upload Dermoscopic Image")
        st.markdown(
            "<p style='color: #aaa; margin-bottom: 1.5rem;'>Upload a high-quality dermoscopic image for AI-powered analysis</p>",
            unsafe_allow_html=True
        )
        
        uploaded_file = st.file_uploader(
            "Choose a dermoscopic image file",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image captured using a dermatoscope for best results.",
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            try:
                # Display image and results using columns
                col_img, col_pred = st.columns([1, 1], gap="large")

                with col_img:
                    st.markdown("### 🖼️ Step 2: Image Preview")
                    image = Image.open(uploaded_file)
                    st.image(image, use_container_width=True)
                    st.caption(
                        f"**Original Size:** {image.size[0]} × {image.size[1]} px | "
                        f"**Model Input:** {IMG_SIZE} × {IMG_SIZE} px"
                    )

                with col_pred:
                    st.markdown("### 🔍 Step 3: AI Analysis Result")

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

                    # Display top prediction with enhanced styling
                    st.markdown(
                        f"""
                        <div style="padding: 1rem; background: rgba(78, 205, 196, 0.1); 
                                    border-radius: 12px; border-left: 4px solid {CLASS_INFO[top_class]['color']}; 
                                    margin: 1rem 0;">
                            <p style="margin: 0 0 0.5rem 0; color: #aaa; font-size: 0.9rem;">Predicted Type</p>
                            <h2 style="margin: 0; color: {CLASS_INFO[top_class]['color']}; font-weight: 700;">
                                {CLASS_INFO[top_class]['full_name']}
                            </h2>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f"""
                        <div style="margin: 1rem 0;">
                            <p style="margin: 0 0 0.5rem 0; color: #aaa; font-size: 0.9rem;">Confidence Level</p>
                            <p style="font-size: 2rem; font-weight: 700; color: #4ECDC4; margin: 0;">
                                {top_prob*100:.1f}%
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.progress(float(top_prob))

                    # Clinical recommendations (moved below main prediction)
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("---", unsafe_allow_html=True)
                    st.markdown("### 🩺 Clinical Recommendations")
                    
                    if risk_level in ['Critical', 'High']:
                        st.markdown(
                            f"""
                            <div style="background: rgba(244, 67, 54, 0.15); border-left: 4px solid #F44336; 
                                        padding: 1.2rem; border-radius: 12px; margin-top: 1rem;">
                                <h4 style="color: #F44336; margin-top: 0;">⚠️ URGENT: Immediate Attention Required</h4>
                                <p style="color: #F0F2F6; margin-bottom: 0.5rem;">
                                    This lesion shows characteristics of <strong>{CLASS_INFO[top_class]['full_name']}</strong>
                                </p>
                                <ul style="color: #F0F2F6; margin: 0.5rem 0 0 0; padding-left: 1.5rem;">
                                    <li>Schedule a <strong>dermatologist appointment immediately</strong></li>
                                    <li>Early detection is crucial for successful treatment</li>
                                    <li>Do not delay seeking professional medical advice</li>
                                </ul>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    elif risk_level == 'Medium':
                        st.markdown(
                            f"""
                            <div style="background: rgba(255, 193, 7, 0.15); border-left: 4px solid #FFC107; 
                                        padding: 1.2rem; border-radius: 12px; margin-top: 1rem;">
                                <h4 style="color: #FFC107; margin-top: 0;">⚡ MEDIUM PRIORITY: Monitor Closely</h4>
                                <p style="color: #F0F2F6; margin-bottom: 0.5rem;">
                                    This lesion shows characteristics of <strong>{CLASS_INFO[top_class]['full_name']}</strong>
                                </p>
                                <ul style="color: #F0F2F6; margin: 0.5rem 0 0 0; padding-left: 1.5rem;">
                                    <li>Schedule a dermatologist appointment within <strong>1-2 weeks</strong></li>
                                    <li>Monitor the lesion for any changes in size, color, or shape</li>
                                    <li>Take photos for comparison during follow-up visits</li>
                                </ul>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"""
                            <div style="background: rgba(76, 175, 80, 0.15); border-left: 4px solid #4CAF50; 
                                        padding: 1.2rem; border-radius: 12px; margin-top: 1rem;">
                                <h4 style="color: #4CAF50; margin-top: 0;">✓ LOW PRIORITY: Routine Monitoring</h4>
                                <p style="color: #F0F2F6; margin-bottom: 0.5rem;">
                                    This lesion appears to be <strong>{CLASS_INFO[top_class]['full_name']}</strong>
                                </p>
                                <ul style="color: #F0F2F6; margin: 0.5rem 0 0 0; padding-left: 1.5rem;">
                                    <li>Continue regular skin self-examinations</li>
                                    <li>Annual dermatology check-ups are recommended</li>
                                    <li>Report any changes to your healthcare provider</li>
                                </ul>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                # Detailed results tab
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("---", unsafe_allow_html=True)
                
                tab_desc, tab_prob, tab_top3 = st.tabs([
                    "📝 Lesion Description", 
                    "📊 Full Probability Distribution", 
                    "🥇 Top 3 Predictions"
                ])
                
                with tab_desc:
                    st.markdown(f"## {CLASS_INFO[top_class]['full_name']}")
                    st.markdown(
                        f"""
                        <div style="background: rgba(78, 205, 196, 0.1); padding: 1.5rem; 
                                    border-radius: 12px; margin: 1rem 0;">
                            <p style="font-size: 1.1rem; line-height: 1.8; color: #F0F2F6;">
                                {CLASS_INFO[top_class]['description']}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f"""
                        <div style="margin-top: 1.5rem;">
                            <p style="color: #aaa; margin-bottom: 0.5rem;">Associated Risk Level</p>
                            <span style="background: {CLASS_INFO[top_class]['color']}; color: white; 
                                        padding: 0.5rem 1.5rem; border-radius: 20px; font-weight: 600; 
                                        display: inline-block; font-size: 1.1rem;">
                                {risk_level}
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with tab_prob:
                    if show_all_probabilities:
                        st.markdown("## 📊 Probability Distribution Across All Classes")
                        st.markdown(
                            "<p style='color: #aaa; margin-bottom: 1.5rem;'>"
                            "Comprehensive confidence scores for all 8 lesion classifications</p>",
                            unsafe_allow_html=True
                        )
                        fig = create_probability_chart(probabilities, CLASS_NAMES)
                        st.plotly_chart(fig, use_container_width=True, theme=None)
                    else:
                        st.info("💡 Detailed chart disabled. Enable it in the 'App Controls' sidebar.")

                with tab_top3:
                    st.markdown("## 🥇 Top 3 Model Predictions")
                    st.markdown(
                        "<p style='color: #aaa; margin-bottom: 1.5rem;'>"
                        "Ranked predictions with confidence scores and risk assessments</p>",
                        unsafe_allow_html=True
                    )
                    top_3_idx = np.argsort(probabilities)[::-1][:3]
                    cols = st.columns(3, gap="large")
                    for i, idx in enumerate(top_3_idx):
                        class_name = CLASS_NAMES[idx]
                        prob = probabilities[idx]
                        
                        with cols[i]:
                            # Enhanced card styling
                            medal_emoji = ["🥇", "🥈", "🥉"][i]
                            st.markdown(f"""
                            <div style="padding: 1.5rem; border-radius: 12px; 
                                        border-left: 5px solid {CLASS_INFO[class_name]['color']}; 
                                        background: linear-gradient(135deg, rgba(30, 30, 30, 0.9) 0%, rgba(40, 40, 40, 0.9) 100%);
                                        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
                                        height: 100%;">
                                <h4 style="margin-top: 0; color: {CLASS_INFO[class_name]['color']} !important; 
                                           font-size: 1.3rem;">
                                    {medal_emoji} #{i+1}: {CLASS_INFO[class_name]['full_name']}
                                </h4>
                                <div style="margin: 1rem 0;">
                                    <p style="color: #aaa; margin: 0; font-size: 0.9rem;">Confidence</p>
                                    <p style="color: #4ECDC4; font-size: 1.8rem; font-weight: 700; margin: 0.5rem 0;">
                                        {prob*100:.1f}%
                                    </p>
                                </div>
                                <div style="margin-top: 1rem;">
                                    <p style="color: #aaa; margin: 0; font-size: 0.9rem;">Risk Level</p>
                                    <span style="background: {CLASS_INFO[class_name]['color']}; color: white; 
                                                padding: 0.4rem 1rem; border-radius: 15px; font-weight: 600; 
                                                display: inline-block; margin-top: 0.5rem;">
                                        {CLASS_INFO[class_name]['risk']}
                                    </span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"⚠️ An error occurred while processing the image. Error details: {str(e)}")
                st.info("Please ensure the image is a valid JPG/PNG file and try again.")
        else:
            st.markdown(
                """
                <div style="text-align: center; padding: 3rem 2rem; background: rgba(78, 205, 196, 0.1); 
                            border-radius: 16px; border: 2px dashed rgba(78, 205, 196, 0.3);">
                    <h3 style="color: #4ECDC4; margin-bottom: 1rem;">👆 Upload Dermoscopic Image</h3>
                    <p style="color: #aaa; font-size: 1.1rem; margin-bottom: 1.5rem;">
                        Please upload a dermoscopic image to begin AI-powered analysis
                    </p>
                    <div style="text-align: left; max-width: 600px; margin: 0 auto; background: rgba(0, 0, 0, 0.3); 
                                padding: 1.5rem; border-radius: 12px;">
                        <h4 style="color: #4ECDC4; margin-top: 0;">💡 Tips for Best Results:</h4>
                        <ul style="color: #F0F2F6; line-height: 1.8;">
                            <li>Use <strong>high-quality dermoscopic images</strong> captured with a dermatoscope</li>
                            <li>Ensure the lesion is <strong>clearly visible and well-lit</strong></li>
                            <li>Upload images in <strong>JPG or PNG format</strong></li>
                            <li><strong>Not validated</strong> for subungual or mucosal lesions</li>
                        </ul>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # --- TAB 2: LESION INFO ---
    with tab_info:
        st.markdown("## 📚 Lesion Category Information")
        st.markdown(
            "<p style='color: #aaa; margin-bottom: 2rem; font-size: 1.1rem;'>"
            "Comprehensive guide to the 8 lesion classifications used by this AI model</p>",
            unsafe_allow_html=True
        )
        
        # Use st.expander for a clean, collapsible list
        for key, info in CLASS_INFO.items():
            full_name = info['full_name']
            color = info['color']
            risk = info['risk']
            
            with st.expander(
                f"**{full_name}** ({key.upper()}) - Risk: {risk}", 
                expanded=(risk in ['Critical', 'High'])
            ):
                st.markdown(
                    f"""
                    <div style="padding: 1rem; background: rgba(78, 205, 196, 0.05); 
                                border-radius: 8px; border-left: 4px solid {color};">
                        <h4 style="color: {color}; margin-top: 0;">Clinical Description</h4>
                        <p style="color: #F0F2F6; line-height: 1.8; font-size: 1.05rem;">
                            {info['description']}
                        </p>
                        <div style="margin-top: 1rem;">
                            <p style="color: #aaa; margin-bottom: 0.5rem;">Associated Risk Level</p>
                            <span style="background: {color}; color: white; padding: 0.5rem 1.2rem; 
                                        border-radius: 20px; font-weight: 600; display: inline-block;">
                                {risk}
                            </span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---", unsafe_allow_html=True)
        st.markdown("## 🔬 What is a Dermoscopic Image?")
        st.markdown(
            """
            <div style="background: rgba(78, 205, 196, 0.1); padding: 2rem; border-radius: 12px; 
                        border-left: 4px solid #4ECDC4;">
                <p style="color: #F0F2F6; line-height: 1.8; font-size: 1.1rem; margin: 0;">
                    Dermoscopic images are captured using a <strong>dermatoscope</strong>, a specialized 
                    medical tool that uses magnification and polarized light to examine skin patterns 
                    beneath the surface. This technique enables more accurate diagnoses by revealing 
                    structures not visible to the naked eye, such as pigment networks, vascular patterns, 
                    and other diagnostic features that help distinguish between benign and malignant lesions.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        

    # Footer
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align: center; color: #777; padding: 2rem 0 1rem 0; font-size: 0.9rem;">
            <p style="margin: 0.5rem 0;">
                <strong style="color: #aaa;">Model:</strong> EfficientNet-B4 | 
                Trained on 25,331 ISIC2019 images | 8-class classification
            </p>
            <p style="margin: 0.5rem 0;">
                <strong style="color: #aaa;">Developed by:</strong> Dr Tom Hutchinson, Oxford, England | 
                For educational and research purposes only
            </p>
            <p style="margin-top: 1rem; color: #555; font-size: 0.85rem;">
                © 2024 Skin Scanner AI | Powered by PyTorch & Streamlit
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
