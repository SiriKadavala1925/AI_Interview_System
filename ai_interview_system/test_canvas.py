import streamlit as st
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Canvas Test", layout="wide")
st.title("Canvas Component Test")
st.write("If you can see a white drawing box below and draw on it with your mouse, the component works.")

canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=3,
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=400,
    width=700,
    drawing_mode="freedraw",
    key="test_canvas",
)

if canvas_result.image_data is not None:
    st.success("Canvas is returning image data correctly!")
    st.write("Image data shape:", canvas_result.image_data.shape)
