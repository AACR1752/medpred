import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Set page config (optional)
st.set_page_config(
    page_title="About - My Streamlit App",
    page_icon="ℹ️",
    layout="wide"
)

# Main content
st.title("About")
st.write("This is a multi-page Streamlit application template.")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('data/med_inv_dataset.csv')
    df.columns = df.columns.str.lower()
    # Convert dateofbill to datetime if it's not already
    df['dateofbill'] = pd.to_datetime(df['dateofbill'])
    return df

df = load_data()

# Create tabs for different functionality
tab1, tab2 = st.tabs(["Drug Selection", "Time Series Forecasting"])

with tab1:
    st.header("Drug Information")
    
    # Drug selection dropdown
    selected_drugname = st.selectbox('Select a drug', df['drugname'].unique())
    
    if st.button("Get Information"):
        # Filter data for the selected drug
        drug_data = df[df['drugname'] == selected_drugname]
        
        # Display basic information
        col1, col2, col3 = st.columns(3)
        with col1:
            subcat_value = drug_data['subcat'].iloc[0]
            st.metric("Subcategory", subcat_value)
        
        with col2:
            total_quantity = drug_data['quantity'].sum()
            st.metric("Total Quantity", f"{total_quantity:,}")
        
        with col3:
            total_returns = drug_data['returnquantity'].sum()
            st.metric("Total Returns", f"{total_returns:,}")
        
        # Show recent transactions
        st.subheader("Recent Transactions")
        st.dataframe(drug_data.sort_values('dateofbill', ascending=False).head(10))

with tab2:
    st.header("Time Series Forecasting")
    
    # Options for forecasting
    forecast_col1, forecast_col2 = st.columns(2)
    
    with forecast_col1:
        # Filter options
        filter_option = st.radio(
            "Filter data for forecasting:",
            ["All Data", "Specific Drug"]
        )
        
        if filter_option == "Specific Drug":
            forecast_drug = st.selectbox('Select drug for forecasting', df['drugname'].unique())
            filtered_df = df[df['drugname'] == forecast_drug]
        else:
            filtered_df = df
    
    with forecast_col2:
        # Fixed SARIMA parameters display
        st.subheader("Model Parameters (Fixed)")
        st.info("Using SARIMA(1,1,1)x(1,1,1,12) model")
        
        # Just show test size option
        test_size = st.slider("Test Set Size (%)", 10, 40, 20) / 100
    
    # Process data and run forecast when button is clicked
    if st.button("Generate Forecast"):
        with st.spinner("Processing data and generating forecast..."):
            # Prepare time series data
            df_summed = filtered_df.groupby('dateofbill')[['quantity', 'returnquantity']].sum().reset_index()
            df_sorted = df_summed.sort_values(by='dateofbill')
            
            # Check if we have enough data
            if len(df_sorted) < 5:  # Arbitrary minimum, adjust as needed
                st.error("Not enough data points for forecasting. Try selecting 'All Data' or a different drug with more data.")
                st.stop()
                
            df_sorted = df_sorted.set_index('dateofbill')
            
            # Calculate net quantity
            df_sorted['net_cumulative'] = (df_sorted['quantity'] - df_sorted['returnquantity'])
            
            # Display the processed data
            st.subheader("Processed Time Series Data")
            st.dataframe(df_sorted.tail())
            
            # Handle train-test split safely
            try:
                # Calculate minimum test size that ensures at least 1 sample
                min_test_size = 1 / len(df_sorted)
                actual_test_size = max(min_test_size, test_size)
                
                # Ensure we have at least 3 samples for training
                actual_train_size = 1 - actual_test_size
                if actual_train_size * len(df_sorted) < 3:
                    actual_train_size = (len(df_sorted) - 3) / len(df_sorted)
                    actual_test_size = 1 - actual_train_size
                
                # Manual split instead of train_test_split
                net_series = df_sorted['net_cumulative']
                split_idx = int(len(net_series) * (1 - actual_test_size))
                train_data = net_series.iloc[:split_idx]
                test_data = net_series.iloc[split_idx:]
                
                if len(train_data) < 3 or len(test_data) < 1:
                    st.warning("Not enough data for a reliable split. Using 70% for training and 30% for testing.")
                    split_idx = int(len(net_series) * 0.7)
                    train_data = net_series.iloc[:split_idx]
                    test_data = net_series.iloc[split_idx:]
                
                st.info(f"Data split: {len(train_data)} training samples, {len(test_data)} test samples")
                
                # Verify we have enough data for seasonal model
                if len(train_data) < 13:  # Need at least one full seasonal cycle + 1
                    st.warning("Not enough data for seasonal modeling. Using non-seasonal ARIMA(1,1,1) instead.")
                    model = SARIMAX(
                        train_data, 
                        order=(1, 1, 1), 
                        seasonal_order=(0, 0, 0, 0)
                    )
                else:
                    # Create and fit the SARIMA model with fixed parameters
                    model = SARIMAX(
                        train_data, 
                        order=(1, 1, 1), 
                        seasonal_order=(1, 1, 1, 12)
                    )
                
                results = model.fit(disp=False)
                
                # Generate predictions
                predictions = results.get_forecast(steps=len(test_data))
                predicted_values = predictions.predicted_mean
                
                # Calculate error metrics
                mse = mean_squared_error(test_data, predicted_values)
                rmse = np.sqrt(mse)
                
                # Display metrics
                st.subheader("Model Performance")
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
                with metric_col2:
                    st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")
                
                # Plot results
                st.subheader("Forecast Results")
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Plot training data
                ax.plot(train_data.index, train_data.values, 'b-', label='Training Data')
                
                # Plot test data and predictions
                ax.plot(test_data.index, test_data.values, 'g-', label='Actual Test Data')
                ax.plot(test_data.index, predicted_values, 'r--', label='Forecast')
                
                # Add confidence intervals if available
                try:
                    pred_ci = predictions.conf_int(alpha=0.05)
                    ax.fill_between(
                        test_data.index, 
                        pred_ci.iloc[:, 0], 
                        pred_ci.iloc[:, 1], 
                        color='pink', 
                        alpha=0.2
                    )
                except:
                    st.info("Confidence intervals could not be computed")
                
                ax.set_title('Time Series Forecast')
                ax.set_xlabel('Date')
                ax.set_ylabel('Net Quantity')
                ax.legend()
                
                # Display the plot
                st.pyplot(fig)
                
                # Future forecast option
                if st.checkbox("Generate Future Forecast"):
                    future_steps = st.slider("Number of future time periods to forecast", 1, 12, 3)
                    
                    # Fit model on all data
                    if len(df_sorted) < 13:
                        full_model = SARIMAX(
                            df_sorted['net_cumulative'],
                            order=(1, 1, 1),
                            seasonal_order=(0, 0, 0, 0)
                        )
                    else:
                        full_model = SARIMAX(
                            df_sorted['net_cumulative'],
                            order=(1, 1, 1),
                            seasonal_order=(1, 1, 1, 12)
                        )
                    
                    full_results = full_model.fit(disp=False)
                    
                    # Generate future forecast
                    future_forecast = full_results.get_forecast(steps=future_steps)
                    future_mean = future_forecast.predicted_mean
                    
                    # Plot future forecast
                    st.subheader("Future Forecast")
                    fig2, ax2 = plt.subplots(figsize=(12, 6))
                    
                    # Plot historical data
                    ax2.plot(df_sorted.index, df_sorted['net_cumulative'], 'b-', label='Historical Data')
                    
                    # Plot forecast
                    ax2.plot(future_mean.index, future_mean, 'r--', label='Future Forecast')
                    
                    try:
                        future_ci = future_forecast.conf_int(alpha=0.05)
                        ax2.fill_between(
                            future_mean.index,
                            future_ci.iloc[:, 0],
                            future_ci.iloc[:, 1],
                            color='red',
                            alpha=0.2
                        )
                        
                        # Display forecast values in a table
                        forecast_df = pd.DataFrame({
                            'Date': future_mean.index,
                            'Forecast': future_mean.values,
                            'Lower CI': future_ci.iloc[:, 0].values,
                            'Upper CI': future_ci.iloc[:, 1].values
                        })
                        st.dataframe(forecast_df.set_index('Date'))
                    except:
                        # Display forecast values without CI
                        forecast_df = pd.DataFrame({
                            'Date': future_mean.index,
                            'Forecast': future_mean.values
                        })
                        st.dataframe(forecast_df.set_index('Date'))
                    
                    ax2.set_title('Future Forecast')
                    ax2.set_xlabel('Date')
                    ax2.set_ylabel('Net Quantity')
                    ax2.legend()
                    
                    st.pyplot(fig2)
                    
            except Exception as e:
                st.error(f"Error in forecasting: {str(e)}")
                st.info("Try using 'All Data' option or a drug with more historical data")

# Sidebar
with st.sidebar:
    st.title("Navigation")
    st.info("Use the tabs above to navigate between different features of the application")
    
    st.subheader("Model Information")
    st.write("This application uses a SARIMA(1,1,1)x(1,1,1,12) model for forecasting.")
    st.write("For small datasets, a simpler ARIMA(1,1,1) model is used.")
    
    # Add sidebar footer
    st.sidebar.markdown("---")
    st.sidebar.info("© 2025 Mainkar Chipa")
    st.sidebar.info("© 2025 Mainkar Chipa")