import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from gurobipy import Model, GRB
from sklearn.ensemble import RandomForestRegressor

# Set page configuration

seed = 800
np.random.seed(seed)

st.set_page_config(
    page_title="Medical Inventory Forecasting",
    page_icon="📈",
    layout="wide"
)

# Main content area
st.title("Medical Inventory Forecasting")
st.write("Analyze and predict inventory trends for medical supplies.")

df = pd.read_csv('data/med_inv_dataset.csv')

df.columns = df.columns.str.lower()
df = df.dropna()
df['dateofbill'] = pd.to_datetime(df['dateofbill'])
df['month_name'] = df['dateofbill'].dt.strftime('%B') # Extract month name
df['month_number'] = df['dateofbill'].dt.month  # Extract month number
df['week_number'] = df['dateofbill'].dt.isocalendar().week  # Extract week number

# Create a bi-weekly period column
df['bi_weekly'] = (df['dateofbill'].dt.day - 1) // 14 + 1

# Group by drug, subcat, month_name, month_number, and bi-weekly period
df_bi_weekly = df.groupby(['subcat', 'month_name', 'month_number', 'bi_weekly'], as_index=False).agg(
    {
        'quantity': 'sum',
        'returnquantity': 'sum',
        'final_cost': 'sum',
        'final_sales': 'sum',
        'rtnmrp': 'sum'
    }
)

# Step 1: Collect top 5 subcategories with the highest sum of quantity
top_5_subcats = df_bi_weekly.groupby('subcat')['quantity'].sum().nlargest(5).index

# Step 2: Filter the dataframe for only the top 5 subcategories
filtered_top_5_per_subcat = df_bi_weekly[df_bi_weekly['subcat'].isin(top_5_subcats)]

# filtered_df_bi_weekly = df_bi_weekly.merge(filtered_top_5_per_subcat[['subcat']], on=['subcat'])
filtered_df_bi_weekly = filtered_top_5_per_subcat.sort_values(by=['subcat', 'month_number', 'bi_weekly'])
# Add a biweekly index for every drugname in every subcat
filtered_df_bi_weekly['biweekly_index'] = (
    filtered_df_bi_weekly.groupby(['subcat'])
    .cumcount() + 1
)

# Filter rows where month_number is 4, 5, or 6 (2 and 3 is added just for their history)
filtered_months_df = filtered_df_bi_weekly[filtered_df_bi_weekly['month_number'].isin([2, 3, 4, 5, 6])]

# The rest of the DataFrame but will remove these months later
rest_df = filtered_df_bi_weekly

def calculate_last_three_cycles(df, subcat, horizon = 'biweekly_index', quanity= 'quantity', train=False):
    ml_df = df[(df['subcat'] == subcat)][[horizon,'subcat', quanity]]

    ml_df['quantity_lastcycle']=ml_df[quanity].shift(+1)
    ml_df['quantity_2cycleback']=ml_df[quanity].shift(+2)
    ml_df['quantity_3cycleback']=ml_df[quanity].shift(+3)
    ml_df['quantity_4cycleback']=ml_df[quanity].shift(+4)
    ml_df['quantity_5cycleback']=ml_df[quanity].shift(+5)

    if train:
        ml_df = ml_df[~ml_df[horizon].isin([9, 10, 11, 12, 13, 14, 15, 16, 17])]

    ml_df = ml_df.dropna() #dropping na is necessary to avoid model failure other option

    X = ml_df[['subcat', horizon,'quantity_lastcycle', 'quantity_2cycleback', 'quantity_3cycleback', 'quantity_4cycleback', 'quantity_5cycleback']]
    y = ml_df[['subcat', quanity]]
    return X, y

trainX = pd.DataFrame(columns=['subcat', 'quantity_lastcycle', 'quantity_2cycleback', 'quantity_3cycleback', 
                               'quantity_4cycleback','quantity_5cycleback'])
trainY = pd.DataFrame(columns=['subcat', 'quantity'])

trainX_rtn = pd.DataFrame(columns=['subcat', 'quantity_lastcycle', 'quantity_2cycleback', 'quantity_3cycleback', 
                               'quantity_4cycleback','quantity_5cycleback'])
trainY_rtn = pd.DataFrame(columns=['subcat', 'returnquantity'])

list_subcat = filtered_top_5_per_subcat['subcat'].unique().tolist()

subcat_dict = {
    list_subcat[0]: {'Capacity': 200, 'shelf_life': (3,8), 'unit_cost': (40.85,322.27), 'salvage_value': (1,617.76)},
    list_subcat[1]: {'Capacity': 400, 'shelf_life': (2,5), 'unit_cost': (40.00,3178.00), 'salvage_value': (1,8014.0)},
    list_subcat[2]: {'Capacity': 100, 'shelf_life': (1,4), 'unit_cost': (40.00,3719.00), 'salvage_value': (1,4462.8)},
    list_subcat[3]: {'Capacity': 100, 'shelf_life': (1,3), 'unit_cost': (42.95,594.95), 'salvage_value': (1,327.11)},
    list_subcat[4]: {'Capacity': 300, 'shelf_life': (5,10), 'unit_cost': (40.00,3491.09), 'salvage_value': (1,1226.0)}
}

# st.write(subcat_dict)

for key in list_subcat:
    X, y = calculate_last_three_cycles(rest_df, subcat = key, quanity = 'quantity', train=True)
    X_rtn, y_rtn = calculate_last_three_cycles(rest_df, subcat = key, quanity = 'returnquantity', train=True)
    trainX = pd.concat([X, trainX])
    trainY = pd.concat([y, trainY])
    trainX_rtn = pd.concat([X_rtn, trainX_rtn])
    trainY_rtn = pd.concat([y_rtn, trainY_rtn])

# st.write(trainX)
# st.write(trainY_rtn)

testX = pd.DataFrame(columns=['subcat', 'quantity_lastcycle', 'quantity_2cycleback', 'quantity_3cycleback',
                              'quantity_4cycleback','quantity_5cycleback'])
testY = pd.DataFrame(columns=['subcat', 'quantity'])

testX_rtn = pd.DataFrame(columns=['subcat', 'quantity_lastcycle', 'quantity_2cycleback', 'quantity_3cycleback', 
                               'quantity_4cycleback','quantity_5cycleback'])
testY_rtn = pd.DataFrame(columns=['subcat', 'returnquantity'])

for key in list_subcat:
    X, y = calculate_last_three_cycles(filtered_months_df, subcat = key, quanity = 'quantity')
    X_rtn, y_rtn = calculate_last_three_cycles(filtered_months_df, subcat = key, quanity = 'returnquantity')
    testX = pd.concat([X, testX])
    testY = pd.concat([y, testY])
    testX_rtn = pd.concat([X_rtn, testX_rtn])
    testY_rtn = pd.concat([y_rtn, testY_rtn])

# st.write(testX)
# st.write(testX_rtn)

selected_subcat = st.selectbox('Select a category', list_subcat)

filtered_df = trainX[trainX['subcat'] == selected_subcat]
# selected_drugname = st.selectbox('Select a drug', filtered_df['drugname'].unique())

# Cost Coefficients
holding_cost = st.slider("Holding Cost", 0.1, 4.0, 2.0)  # Cost per unit held
stockout_penalty = st.slider("Stockout Penalty", 10, 100, 50)  # Cost per stockout
waste_penalty = st.slider("Waste Penalty", 1, 40, 10) # Cost for expired stock

optimize = st.button("Optimize")

if optimize:
    model = RandomForestRegressor(random_state=seed)
    model_rtn = RandomForestRegressor(random_state=seed)

    # This is for demand
    tempx = trainX[(trainX['subcat'] == selected_subcat)].copy()
    t_x = tempx[['quantity_lastcycle', 'quantity_2cycleback', 'quantity_3cycleback','quantity_4cycleback','quantity_5cycleback']]
    t_y = trainY[(trainY['subcat']==selected_subcat)]['quantity']
    model.fit(t_x, t_y)
    test = testX[(testX['subcat'] == selected_subcat)][['quantity_lastcycle', 'quantity_2cycleback', 'quantity_3cycleback',
                                                                                                'quantity_4cycleback','quantity_5cycleback']]
    predicted_demand = model.predict(test)
    predicted_demand = np.ceil(predicted_demand).astype(int) 

    # This is for return qty
    tempx_rtn = trainX_rtn[(trainX_rtn['subcat'] == selected_subcat)].copy()
    t_x = tempx_rtn[['quantity_lastcycle', 'quantity_2cycleback', 'quantity_3cycleback','quantity_4cycleback','quantity_5cycleback']]
    t_y = trainY_rtn[(trainY_rtn['subcat']==selected_subcat)]['returnquantity']

    model_rtn.fit(t_x, t_y)
    
    test = testX_rtn[(testX_rtn['subcat'] == selected_subcat)][['quantity_lastcycle', 'quantity_2cycleback', 'quantity_3cycleback',
                                                                                                'quantity_4cycleback','quantity_5cycleback']]
    predicted_return = model_rtn.predict(test)
    predicted_return = np.ceil(predicted_return).astype(int) 
    
    test_index = testX[(testX['subcat'] == selected_subcat)]['biweekly_index']
    
    T = len(predicted_demand)-1

    temp_dict = subcat_dict[selected_subcat]

    # Create a DataFrame with biweekly_index and predicted_demand
    prediction_df = pd.DataFrame({
        'biweekly_index': test_index,
        'Predicted_Demand': predicted_demand,
        'Return_Prediction': predicted_return,
        "Unit_Cost": np.random.uniform(temp_dict['unit_cost'][0], temp_dict['unit_cost'][1], len(predicted_demand)),  
        "Salvage_Value": np.random.uniform(temp_dict['salvage_value'][0], temp_dict['salvage_value'][1], len(predicted_demand)),
        "Shelf_Life": np.random.randint(temp_dict['shelf_life'][0], temp_dict['shelf_life'][1], len(predicted_demand))  
    })

    prediction_df = prediction_df.reset_index(drop=True)

    # Gurobi Model
    model = Model("Multi_Period_Medical_Inventory_Optimization")

    # Decision Variables
    Q = model.addVars(prediction_df.index, vtype=GRB.CONTINUOUS, name="OrderQty")  # Order quantity

    # State Variables
    I = model.addVars(prediction_df.index, vtype=GRB.CONTINUOUS, name="Inventory")  # Inventory level

    # Auxiliary Variables (Derived)
    Y = model.addVars(prediction_df.index, vtype=GRB.CONTINUOUS, name="Expired")   # Expired stock
    S = model.addVars(prediction_df.index, vtype=GRB.CONTINUOUS, name="Stockout")  # Stockout

    # Objective: Minimize Total Cost
    model.setObjective(
        sum(prediction_df.loc[i, "Unit_Cost"] * Q[i] + 
            holding_cost * (I[i]) +
            stockout_penalty * S[i] + 
            waste_penalty * Y[i] - 
            prediction_df.loc[i, "Salvage_Value"] * prediction_df.loc[i, "Return_Prediction"]
            for i in prediction_df.index),
        GRB.MINIMIZE
    )

    # Constraints:
    for i in prediction_df.index:
        # Safety Stock Constraint
        safety_stock = 0.2 * prediction_df["Predicted_Demand"]  # Example: 20% of demand as buffer
        
        # Inventory Balance Constraint
        if i >= T:  # Ensure we don't reference out-of-bounds indices
            continue
        
        # Ensure inventory level meets safety stock requirements
        model.addConstr(I[i] >= safety_stock[i], name=f"SafetyStock_{i}")
        
        model.addConstr(I[i+1] == I[i] + Q[i] + prediction_df.loc[i, "Return_Prediction"] - prediction_df.loc[i, "Predicted_Demand"] - Y[i], name=f"Inventory_Balance_{i}")

        model.addConstr(I[i+1] <= subcat_dict[selected_subcat]['Capacity'], name=f"Space_constraint{i}")
        
        # Expired Inventory Constraint
        model.addConstr(Y[i] <= I[i], name=f"Expiry_{i}")

        # Stockout Constraint
        model.addConstr(S[i] >= prediction_df.loc[i, "Predicted_Demand"] - I[i] - Q[i], name=f"Stockout_{i}")
    
    # Solve Model
    model.optimize()

    prediction_df["Optimal_Order"] = [Q[i].x for i in prediction_df.index]
    prediction_df["Inventory_Level"] = [I[i].x for i in prediction_df.index]
    prediction_df["Expired_Stock"] = [Y[i].x for i in prediction_df.index]
    prediction_df["Stockouts"] = [S[i].x for i in prediction_df.index]

    tab1, tab2, tab3, tab4 = st.tabs(["Predictions", "Order Quantities", "Inventory Level", "Simulation"])

    st.write("Each cycle represents a bi-weekly period.")

    # Tab 1: Show predictions (Predicted_Demand and Return_Prediction)
    with tab1:
        st.write("### Predictions")
        fig1 = px.line(
        prediction_df,
        x=prediction_df.index,
        y=['Predicted_Demand', 'Return_Prediction'],
        labels={'value': 'Quantity', 'index': 'Cycles'},
        title="Predicted Demand and Return Prediction"
        )
        st.plotly_chart(fig1)


    # Tab 2: Order Quantity tab
    with tab2:
        st.write("### Order Quantity over time") 
        fig3 = px.bar(
            prediction_df,
            x=prediction_df.index,
            y='Optimal_Order',
            labels={'Optimal_Order': 'Order Quantity', 'index': 'Cycles'},
            title="Inventory Level Over Time"
        )
        st.plotly_chart(fig3)
    
    
    
    # Tab 3: Show Inventory_Level over time
    with tab3:
        st.write("### Inventory Level Over Time")
        fig2 = px.line(
            prediction_df,
            x=prediction_df.index,
            y='Inventory_Level',
            labels={'Inventory_Level': 'Inventory Level', 'index': 'Cycles'},
            title="Inventory Level Over Time"
        )
        st.plotly_chart(fig2)

    # Tab 4: Show the entire DataFrame
    with tab4:
        st.write("### Prediction Simulation")
        st.dataframe(prediction_df)
