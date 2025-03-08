import streamlit as st
import pandas as pd

    # Set page configuration
st.set_page_config(
        page_title="My Streamlit App",
        page_icon="📊",
        layout="wide"
)

    # Main content area
st.title("Welcome to My Streamlit App")
st.write("This is a multi-page Streamlit application template.")
    

df = pd.read_csv('data/med_inv_dataset.csv')
df.columns = df.columns.str.lower()
    
# st.selectbox('Select a column', df['drugname'].unique())
selected_drugname = st.selectbox('Select a column', df['drugname'].unique())

if st.button("Predict"):
    subcat_value = df.loc[df['drugname'] == selected_drugname, 'subcat'].iloc[0]
    st.write(f"Sub Cat: {subcat_value}")


    
    
    
    
    
    
    
    # Sidebar (still useful for page-specific controls)
with st.sidebar:
    st.title("Home Page")
    st.info("Select different pages from the sidebar navigation menu above")
        
        # Add sidebar footer
    st.sidebar.markdown("---")
    st.sidebar.info("© 2025 My Streamlit App")
