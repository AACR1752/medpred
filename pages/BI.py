import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# Set page config (optional)
st.set_page_config(
    page_title="Business Intelligence",
    page_icon="📊",
    layout="wide"
)

# Load data

def load_data():
    df = pd.read_csv('data/med_inv_dataset.csv')
    df.columns = df.columns.str.lower()
    # Convert dateofbill to datetime if it's not already
    df['dateofbill'] = pd.to_datetime(df['dateofbill'])
    return df

df = load_data()

st.header("Category Information")

# Drug selection dropdown
selected_catname = st.selectbox('Select a cat', df['subcat'].unique())

df_agg = df.groupby(['subcat', 'dateofbill']).agg(
    total_quantity=('quantity', 'sum'),
    total_returns=('returnquantity', 'sum'),
    total_cost=('final_cost', 'sum'),
    total_revenue=('final_sales', 'sum'),
    total_salvage=('rtnmrp', 'sum')
).reset_index()
# Reindex using 'dateofbill'
df_agg = df_agg.set_index('dateofbill')

# Optional: Sort the index if needed
df_agg = df_agg.sort_index()


if st.button("Get Information"):
    # Filter data for the selected drug
    drug_data = df_agg[df_agg['subcat'] == selected_catname]
    
    # Display basic information
    col1, col2, col3 = st.columns(3)
    with col1:
        subcat_value = drug_data['subcat'].iloc[0]
        st.metric("Subcategory", subcat_value)
    
    with col2:
        total_quantity = drug_data['total_quantity'].sum()
        st.metric("Total Quantity", f"{total_quantity:,}")
    
    with col3:
        total_returns = drug_data['total_returns'].sum()
        st.metric("Total Returns", f"{total_returns:,}")
    
    tab1, tab2 = st.tabs(["Quantities", "Sales"])
    
    with tab1:
        # Plot total_quantity and total_returns over time
        fig = px.line(
            drug_data.reset_index(),  # Reset index to access 'dateofbill' as a column
            x='dateofbill',
            y=['total_quantity', 'total_returns'],
            labels={'value': 'Quantity', 'dateofbill': 'Date'},
            title="Total Quantity and Total Returns Over Time"
        )

        # Display the chart in Streamlit
        st.plotly_chart(fig)

    with tab2:
        # Plot total_cost and total_revenue over time
        fig = px.line(
            drug_data.reset_index(),  # Reset index to access 'dateofbill' as a column
            x='dateofbill',
            y=['total_cost', 'total_revenue'],
            labels={'value': 'Cost / Revenue', 'dateofbill': 'Date'},
            title="Total Cost and Total Revenue Over Time"
        )

        # Display the chart in Streamlit
        st.plotly_chart(fig)
    
    # Show recent transactions
    st.subheader("Recent Transactions")
    st.dataframe(drug_data.sort_values('dateofbill', ascending=False).head(10))
