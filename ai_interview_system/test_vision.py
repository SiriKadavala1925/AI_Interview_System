"""
Standalone test: draws a simple diagram, saves it as PNG,
then asks the moondream vision model to describe it.
Run this BEFORE touching app.py, to confirm the vision
pipeline works on this machine.
"""
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
import ollama
import io

st.set_page_config(page_title="Vision Model Test", layout="wide")
st.title("🧪 Vision Model Test — Moondream2")
st.write("Draw a simple diagram below (e.g. 3 boxes connected by arrows), then click Analyze.")

canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.1)",
    stroke_width=3,
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=350,
    width=700,
    drawing_mode="freedraw",
    key="vision_test_canvas",
)

if st.button("🔍 Analyze Drawing with Moondream2"):
    if canvas_result.image_data is None:
        st.error("Please draw something first.")
    else:
        with st.spinner("Moondream2 is analyzing your drawing..."):
            # Convert canvas RGBA numpy array to a PIL image, then to PNG bytes
            img_array = canvas_result.image_data.astype(np.uint8)
            pil_image = Image.fromarray(img_array, mode="RGBA")

            # Composite onto white background (canvas has transparent areas)
            background = Image.new("RGBA", pil_image.size, (255, 255, 255, 255))
            composited = Image.alpha_composite(background, pil_image).convert("RGB")

            # Save to temp file for Ollama
            temp_path = "temp_vision_test.png"
            composited.save(temp_path)

            try:
                response = ollama.generate(
                    model="moondream",
                    prompt="Describe what is drawn in this image. Focus on shapes, boxes, arrows, and any structure you see.",
                    images=[temp_path]
                )
                description = response["response"].strip()

                st.success("✅ Moondream2 responded successfully!")
                st.markdown("**Model's description of your drawing:**")
                st.info(description)

            except Exception as e:
                st.error(f"❌ Error calling Moondream2: {e}")