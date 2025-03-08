import streamlit as st
import numpy as np
import pandas as pd

# Set page config (optional, can be omitted if you want to inherit from Home.py)
st.set_page_config(
    page_title="Data Visualization - My Streamlit App",
    page_icon="📈",
    layout="wide"
)

# Main content
st.title("Data Visualization")
st.write("This page contains your data visualizations.")

# Create tabs for different visualizations
tab1, tab2 = st.tabs(["Chart View", "Data View"])

with tab1:
    st.subheader("Chart Visualization")
    
    # Example chart data (replace with your actual data)
    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['A', 'B', 'C']
    )
    
    # Display chart
    st.line_chart(chart_data)
    
with tab2:
    st.subheader("Data Table")
    # Show the underlying data
    st.dataframe(chart_data)

# Sidebar for this specific page
with st.sidebar:
    st.title("Visualization Controls")
    
    # Add filters or controls specific to this page
    st.subheader("Filters")
    date_range = st.date_input("Select date range")
    category = st.multiselect("Select categories", ["Category A", "Category B", "Category C"])
    show_details = st.checkbox("Show advanced options", value=False)
    
    if show_details:
        chart_type = st.radio("Chart type", ["Line", "Bar", "Area"])
        smoothing = st.slider("Smoothing factor", 0.0, 1.0, 0.2)
    
    # Add sidebar footer
    st.sidebar.markdown("---")
    st.sidebar.info("© 2025 My Streamlit App")