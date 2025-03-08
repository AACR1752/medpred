import streamlit as st

def main():
    # Set page configuration
    st.set_page_config(
        page_title="My Streamlit App",
        page_icon="📊",
        layout="wide"
    )

    # Sidebar
    with st.sidebar:
        st.title("Navigation")
        
        # Add some sidebar components
        option = st.selectbox(
            "Choose a section",
            ["Home", "Data Visualization", "About"]
        )
        
        st.divider()
        
        # Add some filters or controls
        if option == "Data Visualization":
            st.subheader("Filters")
            date_range = st.date_input("Select date range")
            category = st.multiselect("Select categories", ["Category A", "Category B", "Category C"])
            show_details = st.checkbox("Show details", value=True)
        
        # Add sidebar footer
        st.sidebar.markdown("---")
        st.sidebar.info("© 2025 My Streamlit App")

    # Main content area
    if option == "Home":
        st.title("Welcome to My Streamlit App")
        st.write("This is a simple Streamlit application template with a sidebar.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Features")
            st.write("✅ Responsive layout")
            st.write("✅ Interactive components")
            st.write("✅ Easy navigation")
        
        with col2:
            st.subheader("Get Started")
            st.write("Select an option from the sidebar to navigate through the application.")
            
            if st.button("Show Example"):
                st.success("This is an example of a button action!")
                
    elif option == "Data Visualization":
        st.title("Data Visualization")
        st.write("This section would contain your data visualizations.")
        
        # Example chart (replace with your actual visualization code)
        import numpy as np
        import pandas as pd
        
        chart_data = pd.DataFrame(
            np.random.randn(20, 3),
            columns=['A', 'B', 'C']
        )
        
        st.line_chart(chart_data)
        
        if 'show_details' in locals() and show_details:
            st.subheader("Data Details")
            st.dataframe(chart_data)
            
    elif option == "About":
        st.title("About")
        st.write("This is a template for a Streamlit application with a sidebar for navigation.")
        st.write("You can customize this template to create your own application.")
        
        st.info("For more information, visit [Streamlit Documentation](https://docs.streamlit.io)")

if __name__ == "__main__":
    main()