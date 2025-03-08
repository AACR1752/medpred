import streamlit as st

    # Set page configuration
st.set_page_config(
        page_title="My Streamlit App",
        page_icon="📊",
        layout="wide"
)

    # Main content area
st.title("Welcome to My Streamlit App")
st.write("This is a multi-page Streamlit application template.")
    

    
    # Sidebar (still useful for page-specific controls)
with st.sidebar:
    st.title("Home Page")
    st.info("Select different pages from the sidebar navigation menu above")
        
        # Add sidebar footer
    st.sidebar.markdown("---")
    st.sidebar.info("© 2025 My Streamlit App")
