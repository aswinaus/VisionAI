"""
Streamlit web application for Vision AI Showcase.
"""

import streamlit as st
import numpy as np
from PIL import Image
import io
import json
from pathlib import Path
import time

from src.models.registry import ModelRegistry
from src.inference.pipeline import InferencePipeline
from src.data.loaders import ImageDataLoader
from src.utils.config import ConfigManager


@st.cache_resource
def load_model_registry():
    """Load and cache model registry."""
    registry = ModelRegistry("models")
    if len(registry) == 0:
        registry.create_default_models()
    return registry


@st.cache_data
def load_sample_images():
    """Load sample images for demonstration."""
    sample_dir = Path("data/samples")
    if not sample_dir.exists():
        return []
    
    loader = ImageDataLoader(str(sample_dir))
    if len(loader.image_paths) > 0:
        return [str(path) for path in loader.image_paths[:10]]  # Limit to first 10
    return []


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Vision AI Showcase",
        page_icon="👁️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("👁️ Vision AI Showcase")
    st.markdown("Demonstrate computer vision capabilities with various AI models")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Load model registry
    try:
        registry = load_model_registry()
        available_models = registry.get_available_models()
        
        if not available_models:
            st.error("No models available. Please run the setup first.")
            st.code("python main.py --mode setup")
            return
        
        # Model selection
        selected_model = st.sidebar.selectbox(
            "Select Model",
            available_models,
            index=0
        )
        
        # Display model info
        model_info = registry.get_model_info(selected_model)
        with st.sidebar.expander("Model Information"):
            st.json(model_info["config"])
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Input Image")
        
        # Image upload options
        input_method = st.radio(
            "Choose input method:",
            ["Upload Image", "Use Sample Images", "Camera Input"]
        )
        
        uploaded_image = None
        
        if input_method == "Upload Image":
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['png', 'jpg', 'jpeg', 'bmp', 'webp'],
                help="Upload an image for analysis"
            )
            
            if uploaded_file is not None:
                uploaded_image = Image.open(uploaded_file)
        
        elif input_method == "Use Sample Images":
            sample_images = load_sample_images()
            if sample_images:
                selected_sample = st.selectbox(
                    "Select sample image:",
                    sample_images,
                    format_func=lambda x: Path(x).name
                )
                if selected_sample:
                    uploaded_image = Image.open(selected_sample)
            else:
                st.info("No sample images available. Add images to data/samples/")
        
        elif input_method == "Camera Input":
            camera_image = st.camera_input("Take a picture")
            if camera_image is not None:
                uploaded_image = Image.open(camera_image)
        
        # Display input image
        if uploaded_image is not None:
            st.image(uploaded_image, caption="Input Image", use_column_width=True)
            
            # Image information
            st.write(f"**Image Size:** {uploaded_image.size}")
            st.write(f"**Image Mode:** {uploaded_image.mode}")
    
    with col2:
        st.header("Analysis Results")
        
        if uploaded_image is not None:
            # Process button
            if st.button("🔍 Analyze Image", type="primary"):
                with st.spinner("Processing image..."):
                    try:
                        # Load model
                        model = registry.load_model(selected_model)
                        
                        # Create inference pipeline
                        pipeline = InferencePipeline(model)
                        
                        # Run inference
                        start_time = time.time()
                        result = pipeline.predict_single_pil(uploaded_image)
                        inference_time = time.time() - start_time
                        
                        # Display results
                        st.success(f"Analysis completed in {inference_time:.2f} seconds")
                        
                        # Results based on model type
                        model_type = model_info["config"].get("type", "unknown")
                        
                        if model_type == "classification" or model_type == "custom_classification":
                            st.subheader("Classification Results")
                            
                            # Top prediction
                            top_class = result.get("top_class", "Unknown")
                            top_confidence = result.get("top_confidence", 0.0)
                            
                            st.metric(
                                "Top Prediction",
                                top_class,
                                f"{top_confidence*100:.1f}%"
                            )
                            
                            # All predictions
                            predictions = result.get("predictions", [])
                            if predictions:
                                st.subheader("All Predictions")
                                for pred in predictions[:5]:  # Show top 5
                                    col_class, col_conf = st.columns([3, 1])
                                    with col_class:
                                        st.write(f"**{pred['class_name']}**")
                                    with col_conf:
                                        st.write(f"{pred['percentage']:.1f}%")
                                    
                                    # Progress bar for confidence
                                    st.progress(pred['confidence'])
                        
                        # Raw results in expandable section
                        with st.expander("Raw Results (JSON)"):
                            st.json(result)
                        
                        # Download results
                        result_json = json.dumps(result, indent=2)
                        st.download_button(
                            label="📥 Download Results",
                            data=result_json,
                            file_name=f"analysis_result_{int(time.time())}.json",
                            mime="application/json"
                        )
                        
                    except Exception as e:
                        st.error(f"Error during analysis: {e}")
                        if st.checkbox("Show detailed error"):
                            st.exception(e)
        else:
            st.info("Upload an image or select a sample to start analysis")
    
    # Footer information
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Available Models")
        for model_name in available_models:
            info = registry.get_model_info(model_name)
            description = info["config"].get("description", "No description")
            st.write(f"**{model_name}**: {description}")
    
    with col2:
        st.subheader("Model Statistics")
        if selected_model:
            info = registry.get_model_info(selected_model)
            st.write(f"**Architecture**: {info['config'].get('architecture', 'Custom')}")
            st.write(f"**Classes**: {info['config'].get('num_classes', 'Unknown')}")
            st.write(f"**Input Size**: {info['config'].get('input_size', 'Variable')}")
    
    with col3:
        st.subheader("System Information")
        import torch
        st.write(f"**PyTorch Version**: {torch.__version__}")
        st.write(f"**CUDA Available**: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            st.write(f"**GPU**: {torch.cuda.get_device_name()}")


if __name__ == "__main__":
    main()