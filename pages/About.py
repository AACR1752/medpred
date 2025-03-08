import streamlit as st

# Set page config (optional)
st.set_page_config(
    page_title="About - My Streamlit App",
    page_icon="ℹ️",
    layout="wide"
)

# Main content
st.title("About")
st.write("This is a multi-page Streamlit application template.")

# Create an expandable section with more details
with st.expander("Application Details"):
    st.write("""
    This template demonstrates how to create a multi-page Streamlit application.
    Each page is a separate Python file, making it easier to organize and maintain your code.
    
    The navigation between pages is handled automatically by Streamlit through the sidebar.
    """)

# Team information section
st.subheader("Team")
col1, col2, col3 = st.columns(3)

with col1:
    st.image("https://via.placeholder.com/150", width=150)
    st.markdown("**John Doe**")
    st.markdown("Lead Developer")

with col2:
    st.image("https://via.placeholder.com/150", width=150)
    st.markdown("**Jane Smith**")
    st.markdown("Data Scientist")

with col3:
    st.image("https://via.placeholder.com/150", width=150)
    st.markdown("**Bob Johnson**")
    st.markdown("UX Designer")

# Contact form
st.subheader("Contact Us")
with st.form("contact_form"):
    name = st.text_input("Name")
    email = st.text_input("Email")
    message = st.text_area("Message")
    submitted = st.form_submit_button("Submit")
    
    if submitted:
        st.success("Thank you for your message! We'll get back to you soon.")

# Resources section
st.subheader("Resources")
st.markdown("""
- [Streamlit Documentation](https://docs.streamlit.io)
- [GitHub Repository](https://github.com/yourusername/your-repo)
- [Report an Issue](https://github.com/yourusername/your-repo/issues)
""")

# Sidebar for this specific page
with st.sidebar:
    st.title("About Page")
    st.info("Learn more about our application and team")
    
    # Version information
    st.markdown("**App Version:** 1.0.0")
    
    # Add sidebar footer
    st.sidebar.markdown("---")
    st.sidebar.info("© 2025 My Streamlit App")