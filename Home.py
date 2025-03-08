import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="My Streamlit App",
    page_icon="📊",
    layout="wide"
)

# Main content area
st.title("Welcome to My Streamlit App")
st.write("This is a multi-page Streamlit application template.")

# Load and prepare data
df = pd.read_csv('data/med_inv_dataset.csv')
df.columns = df.columns.str.lower()

# Add 'All Data' option to drug selection
options = ['All Data'] + list(df['drugname'].unique())
selected_drugname = st.selectbox('Select a drug', options)

# Filter data based on selection
if selected_drugname == 'All Data':
    filtered_df = df
else:
    filtered_df = df[df['drugname'] == selected_drugname]

# Display basic info
if st.button("Show Basic Info"):
    if selected_drugname != 'All Data':
        subcat_value = df.loc[df['drugname'] == selected_drugname, 'subcat'].iloc[0]
        st.write(f"Sub Cat: {subcat_value}")
    else:
        st.write(f"Total drugs: {df['drugname'].nunique()}")
        st.write(f"Total subcategories: {df['subcat'].nunique()}")

# Forecasting section
if st.checkbox("Generate Future Forecast"):
    # Check if there's enough data
    if len(filtered_df) < 5:  # Minimum data points needed
        st.error("Not enough data points for forecasting. Try selecting 'All Data' or a different drug with more data.")
    else:
        # Assuming there's a date column and a quantity/inventory column
        # If these don't exist, you'll need to adjust accordingly
        st.write("Generating forecast...")
        
        # For this example, we'll create a simple time series if it doesn't exist
        if 'date' not in filtered_df.columns or 'quantity' not in filtered_df.columns:
            st.warning("No time series data found. Creating sample data for demonstration.")
            # Create sample time series data for demonstration
            dates = pd.date_range(start='1/1/2024', periods=len(filtered_df), freq='D')
            filtered_df['date'] = dates
            if 'quantity' not in filtered_df.columns:
                filtered_df['quantity'] = np.random.randint(10, 100, size=len(filtered_df))
        
        # Set date as index for time series forecasting
        ts_df = filtered_df.set_index('date')['quantity']
        
        # Number of periods to forecast
        forecast_periods = st.slider("Forecast Periods (days)", 7, 90, 30)
        
        try:
            # Fit Exponential Smoothing model
            model = ExponentialSmoothing(ts_df, 
                                        trend='add', 
                                        seasonal='add', 
                                        seasonal_periods=7)
            model_fit = model.fit()
            
            # Generate forecast
            forecast = model_fit.forecast(forecast_periods)
            
            # Plot results
            fig, ax = plt.subplots(figsize=(10, 6))
            ts_df.plot(ax=ax, label='Historical Data')
            forecast.plot(ax=ax, label='Forecast', color='red')
            plt.title(f"Inventory Forecast for {selected_drugname}")
            plt.legend()
            st.pyplot(fig)
            
            # Display forecast data
            st.subheader("Forecast Data")
            st.dataframe(forecast.reset_index().rename(
                columns={'index': 'Date', 0: 'Predicted Quantity'}))
            
        except Exception as e:
            st.error(f"Error generating forecast: {e}")
            st.write("Try selecting a different drug or using 'All Data'")

# Sidebar (still useful for page-specific controls)
with st.sidebar:
    st.title("Home Page")
    st.info("Select different pages from the sidebar navigation menu above")
    
    # Add sidebar footer
    st.sidebar.markdown("---")
    st.sidebar.info("© 2025 Mainkar Chipa")