import streamlit as st
import pandas as pd
import numpy as np
import io
import zipfile
from datetime import datetime
import math
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="LTV Calculation Dashboard",
    page_icon="📊",
    layout="wide"
)

def process_files(uploaded_files):
    """Process uploaded CSV files and combine them into a single dataframe"""
    if not uploaded_files:
        return None
    
    dataframes = []
    
    for uploaded_file in uploaded_files:
        try:
            # Read CSV file
            df = pd.read_csv(uploaded_file, low_memory=False)
            
            # Add source file column
            df['source_file'] = uploaded_file.name
            dataframes.append(df)
            
        except Exception as e:
            st.error(f"Error reading {uploaded_file.name}: {str(e)}")
            return None
    
    if not dataframes:
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Replace hyphens with underscores and convert to lowercase
    combined_df.columns = combined_df.columns.str.replace('-', '_').str.replace(' ', '_').str.lower()
    
    return combined_df

def calculate_product_raw(combined_df):
    """Calculate ProductRaw dataframe"""
    # Ensure purchase_date is datetime
    combined_df['purchase_date'] = pd.to_datetime(combined_df['purchase_date'], utc=True, errors='coerce')
    
    # Remove rows where purchase_date couldn't be parsed
    combined_df = combined_df.dropna(subset=['purchase_date'])
    
    # Filter data
    filtered_df = combined_df[
        (combined_df['currency'] == 'USD') &
        (combined_df['item_price'] > 0) &
        (combined_df['purchase_date'].dt.strftime('%Y-%m-01') > '2022-09-01')
    ].copy()
    
    # Calculate POME date
    filtered_df['buyer_email'] = filtered_df['buyer_email'].fillna('NA')
    filtered_df['pome_date'] = filtered_df.groupby(['buyer_email', 'merchant_sku'], dropna=False)['purchase_date'].transform('min').dt.strftime('%Y-%m-%d')
    
    # Create ntb dataframe
    result_df = filtered_df[['buyer_email', 'merchant_sku', 'pome_date']].drop_duplicates().reset_index(drop=True)
    ntb = result_df.rename(columns={'buyer_email': 'user_id'})
    
    # Create raw_data
    raw_data_df = filtered_df[['buyer_email', 'amazon_order_id', 'merchant_sku', 'purchase_date', 'item_price', 'shipped_quantity']].copy()
    raw_data_df = raw_data_df.rename(columns={
        'buyer_email': 'user_id',
        'amazon_order_id': 'order_id',
        'item_price': 'sales',
        'shipped_quantity': 'quantity'
    })
    
    raw_data_df['purchase_date'] = raw_data_df['purchase_date'].dt.strftime('%Y-%m-%d')
    raw_data_df['sales'] = raw_data_df['sales'].astype(float).round(2)
    raw_data = raw_data_df.drop_duplicates()
    
    # Perform left join and calculate months
    process_data = raw_data.merge(ntb, on=['user_id', 'merchant_sku'], how='left')
    process_data['purchase_date'] = pd.to_datetime(process_data['purchase_date'])
    process_data['pome_date'] = pd.to_datetime(process_data['pome_date'])
    process_data['date_diff'] = (process_data['purchase_date'] - process_data['pome_date']).dt.days
    process_data['months'] = np.ceil(process_data['date_diff'] / 30)
    
    # Group and aggregate
    process_data_agg = process_data.groupby([
        'user_id', 'merchant_sku', 'order_id', 'purchase_date', 'pome_date'
    ], dropna=False).agg({
        'sales': 'sum',
        'quantity': 'sum',
        'date_diff': 'first',
        'months': 'first'
    }).reset_index()
    
    result = process_data_agg.groupby(['merchant_sku', 'months'], dropna=False).agg({
        'user_id': 'nunique',
        'sales': 'sum',
        'quantity': 'sum',
        'order_id': 'nunique'
    }).reset_index()
    
    ProductRaw = result.rename(columns={'user_id': 'users', 'order_id': 'orders'})
    
    return ProductRaw

def calculate_product_summary(combined_df):
    """Calculate ProductSummary dataframe"""
    # Create cte_tbl equivalent
    cte_tbl = combined_df[['buyer_email', 'merchant_sku', 'amazon_order_id', 'purchase_date', 'item_price', 'shipped_quantity', 'currency']].copy()
    cte_tbl['buyer_email'] = cte_tbl['buyer_email'].fillna('NA')
    
    # Apply STRFTIME transformations
    cte_tbl['purch_date'] = pd.to_datetime(cte_tbl['purchase_date']).dt.strftime('%Y-%m-%d')
    cte_tbl['purch_month'] = pd.to_datetime(cte_tbl['purchase_date']).dt.strftime('%Y-%m-01')
    
    # Apply filters
    cte_tbl = cte_tbl[
        (cte_tbl['currency'] == 'USD') &
        (cte_tbl['item_price'] > 0) &
        (cte_tbl['purch_month'] > '2022-09-01')
    ].copy()
    
    # Add magnitude column
    cte_tbl['magnitude'] = cte_tbl.groupby(['buyer_email', 'merchant_sku'], dropna=False)['purchase_date'].rank(method='first', ascending=True)
    
    # Create derived columns
    df = cte_tbl.copy()
    df['buyer_email'] = df['buyer_email'].astype(str)
    
    df['orig_sales'] = np.where(df['magnitude'] == 1, df['item_price'], 0)
    df['orig_quantity'] = np.where(df['magnitude'] == 1, df['shipped_quantity'], 0)
    df['orig_buyer_email_for_count'] = np.where(df['magnitude'] == 1, df['buyer_email'], np.nan)
    
    df['repeat_sales'] = np.where(df['magnitude'] > 1, df['item_price'], 0)
    df['repeat_quantity'] = np.where(df['magnitude'] > 1, df['shipped_quantity'], 0)
    df['repeat_buyer_email_for_count'] = np.where(df['magnitude'] > 1, df['buyer_email'], np.nan)
    
    # Group by and aggregate
    group_cols = ['buyer_email', 'merchant_sku', 'amazon_order_id']
    agg_operations = {
        'orig_users': ('orig_buyer_email_for_count', 'nunique'),
        'orig_sales': ('orig_sales', 'sum'),
        'orig_quantity': ('orig_quantity', 'sum'),
        'repeat_users': ('repeat_buyer_email_for_count', 'nunique'),
        'repeat_sales': ('repeat_sales', 'sum'),
        'repeat_quantity': ('repeat_quantity', 'sum')
    }
    
    cte_tbl_raw = df.groupby(group_cols, as_index=False, dropna=False).agg(**agg_operations)
    
    # Create cte_test equivalent using UNION logic
    cte_test_part1 = cte_tbl[cte_tbl['magnitude'] == 1].groupby('merchant_sku', dropna=False).agg({
        'buyer_email': 'nunique',
        'item_price': 'sum',
        'shipped_quantity': 'sum'
    }).reset_index()
    
    cte_test_part1.columns = ['merchant_sku', 'users', 'sales', 'quantity']
    cte_test_part1['users1'] = 0
    cte_test_part1['sales1'] = 0
    cte_test_part1['quantity1'] = 0
    cte_test_part1['orders'] = cte_test_part1['users']
    cte_test_part1['orders1'] = 0
    
    # Second part: magnitude > 1
    cte_test_part2 = cte_tbl[cte_tbl['magnitude'] > 1].groupby('merchant_sku', dropna=False).agg({
        'buyer_email': 'nunique',
        'item_price': 'sum',
        'shipped_quantity': 'sum'
    }).reset_index()
    
    cte_test_part2.columns = ['merchant_sku', 'users1', 'sales1', 'quantity1']
    cte_test_part2['users'] = 0
    cte_test_part2['sales'] = 0
    cte_test_part2['quantity'] = 0
    cte_test_part2['orders'] = 0
    cte_test_part2['orders1'] = cte_test_part2['users1']
    
    # Combine both parts
    cte_test = pd.concat([cte_test_part1, cte_test_part2], ignore_index=True)
    
    # Final aggregation
    ProductSummary = cte_test.groupby('merchant_sku', dropna=False).agg({
        'users': 'sum',
        'sales': 'sum',
        'quantity': 'sum',
        'users1': 'sum',
        'sales1': 'sum',
        'quantity1': 'sum',
        'orders': 'sum',
        'orders1': 'sum'
    }).reset_index()
    
    ProductSummary['orders'] = ProductSummary['quantity']
    ProductSummary['orders1'] = ProductSummary['quantity']
    
    return ProductSummary

def calculate_raw_data(combined_df):
    """Calculate RawData dataframe"""
    # Create ntb equivalent
    # Convert SQL RawData query to pandas operations

    # Step 1: Create ntb equivalent
    ntb_df = combined_df[combined_df['purchase_date'].dt.strftime('%Y-%m-01') > '2022-09-01'].copy()
    ntb_df['buyer_email'] = ntb_df['buyer_email'].fillna('NA')
    ntb_df['pome_month'] = ntb_df.groupby('buyer_email', dropna=False)['purchase_date'].transform(lambda x: x.dt.strftime('%Y-%m-01').min())
    ntb = ntb_df[['buyer_email', 'pome_month']].drop_duplicates().rename(columns={'buyer_email': 'user_id'})

    # Step 2: Create pntb equivalent  
    pntb_df = combined_df[combined_df['purchase_date'].dt.strftime('%Y-%m-01') > '2022-09-01'].copy()
    pntb_df['pntb_title'] = pntb_df.groupby('merchant_sku', dropna=False)['title'].transform('min')
    pntb_df['pntb_month'] = pntb_df.groupby('merchant_sku', dropna=False)['purchase_date'].transform(lambda x: x.dt.strftime('%Y-%m-01').min())
    pntb = pntb_df[['merchant_sku', 'pntb_title', 'pntb_month']].drop_duplicates().rename(columns={'merchant_sku': 'tracking_id'})

    # Step 3: Create output_tbl equivalent
    # Filter the main dataframe
    filtered_df = combined_df[
        (combined_df['currency'] == 'USD') &
        (combined_df['item_price'] > 0) &
        (combined_df['purchase_date'].dt.strftime('%Y-%m-01') > '2022-09-01')
    ].copy()

    # Create month column
    filtered_df['month'] = filtered_df['purchase_date'].dt.strftime('%Y-%m-01')
    filtered_df['buyer_email'] = filtered_df['buyer_email'].fillna('NA')

    # Left join with ntb
    output_tbl1 = filtered_df.merge(
        ntb,
        left_on='buyer_email',
        right_on='user_id',
        how='left'
    )

    # Left join with pntb
    output_tbl = output_tbl1.merge(
        pntb,
        left_on='merchant_sku', 
        right_on='tracking_id',
        how='left'
    )

    output_tbl['buyer_email'] = output_tbl['buyer_email'].fillna('NA')
    output_tbl['amazon_order_id'] = output_tbl['amazon_order_id'].fillna('NA')


    # Group by and aggregate
    RawData = output_tbl.groupby(['month', 'merchant_sku', 'pntb_title', 'pome_month', 'pntb_month'], dropna=False).agg({
        'buyer_email': 'nunique',       # users
        'shipped_quantity': 'sum',      # quantity  
        'amazon_order_id': 'nunique',   # orders
        'item_price': 'sum'             # sales
    }).reset_index()

    # Rename columns
    RawData = RawData.rename(columns={
        'buyer_email': 'users',
        'shipped_quantity': 'quantity',
        'amazon_order_id': 'orders', 
        'item_price': 'sales'
    })

    # Calculate derived columns (cast as decimal(10,2) equivalent)
    RawData['avg_per_item'] = (RawData['sales'] / RawData['quantity']).round(2)
    RawData['avg_per_order'] = (RawData['sales'] / RawData['orders']).round(2)
    RawData['avg_per_user'] = (RawData['sales'] / RawData['users']).round(2)

    # Round sales to 2 decimal places (equivalent to cast as decimal(10,2))
    RawData['sales'] = RawData['sales'].round(2)
    
    return RawData

def calculate_raw_data_wo_sku(combined_df):
    # Convert SQL RawData query to pandas operations
    # Step 1: Create ntb equivalent
    ntb_df_wo_sku = combined_df[combined_df['purchase_date'].dt.strftime('%Y-%m-01') > '2022-09-01'].copy()
    ntb_df_wo_sku['buyer_email'] = ntb_df_wo_sku['buyer_email'].fillna('NA')
    ntb_df_wo_sku['pome_month'] = ntb_df_wo_sku.groupby('buyer_email', dropna=False)['purchase_date'].transform(lambda x: x.dt.strftime('%Y-%m-01').min())
    ntb_df_wo_sku = ntb_df_wo_sku[['buyer_email', 'pome_month']].drop_duplicates().rename(columns={'buyer_email': 'user_id'})

    # Step 3: Create output_tbl equivalent
    # Filter the main dataframe
    filtered_df_wo_sku = combined_df[
        (combined_df['currency'] == 'USD') &
        (combined_df['item_price'] > 0) &
        (combined_df['purchase_date'].dt.strftime('%Y-%m-01') > '2022-09-01')
    ].copy()

    # Create month column
    filtered_df_wo_sku['month'] = filtered_df_wo_sku['purchase_date'].dt.strftime('%Y-%m-01')
    filtered_df_wo_sku['buyer_email'] = filtered_df_wo_sku['buyer_email'].fillna('NA')

    filtered_df_wo_sku = filtered_df_wo_sku[['month', 'buyer_email']].reset_index(drop=True)

    # Left join with ntb
    output_tbl_wo_sku = filtered_df_wo_sku.merge(
        ntb_df_wo_sku,
        left_on='buyer_email',
        right_on='user_id',
        how='left'
    )

    output_tbl_wo_sku['buyer_email'] = output_tbl_wo_sku['buyer_email'].fillna('NA').drop_duplicates()


    # Group by and aggregate
    RawDataWithoutSku = output_tbl_wo_sku.groupby(['month', 'pome_month'], dropna=False).agg({
        'buyer_email': 'nunique',       # users
    }).reset_index()

    # Rename columns
    RawDataWithoutSku = RawDataWithoutSku.rename(columns={
        'buyer_email': 'users',
    })

    return RawDataWithoutSku
    

def calculate_cohort_analysis(raw_data, selected_merchant_sku=None):
    """Calculate cohort analysis"""
    # Get unique months and filter for last 12
    pome_months = raw_data['pome_month'].unique()
    pome_months_no_nan = [x for x in pome_months if pd.notna(x)]
    all_months = sorted(pome_months_no_nan)[-12:]
    
    # Apply filtering based on merchant SKU selection
    if selected_merchant_sku and selected_merchant_sku != "All":
        raw_data['IsFiltered'] = np.where(raw_data['merchant_sku'] == selected_merchant_sku, 1, 0)
        filter_msg = f"Filtering data for merchant SKU: {selected_merchant_sku}"
    else:
        raw_data['IsFiltered'] = 1
        filter_msg = "Using all merchant SKUs (no filtering applied)"
    
    # Calculate cohort sizes
    cohort_data = []
    for selected_month in all_months:
        filtered_data = raw_data[
            (raw_data['month'] == selected_month) & 
            (raw_data['pome_month'] == selected_month) & 
            (raw_data['IsFiltered'] == 1)
        ]
        cohort_size = filtered_data['users'].sum()
        cohort_data.append({
            'POME Month': selected_month,
            'Cohort Size': cohort_size
        })
    
    cohort_df = pd.DataFrame(cohort_data)
    
    # Create metrics structure
    metrics = ['Active Customers', 'Quantities', 'Orders', 'Revenue', 'Cumulative Revenue', 'Retention Rate', 'LTV']
    month_columns = sorted(all_months)
    
    # Create the base structure
    table_data = []
    for idx, row in cohort_df.iterrows():
        pome_month = row['POME Month']
        cohort_size = row['Cohort Size']
        
        for metric in metrics:
            row_data = {
                'POME Month': pome_month,
                'Cohort Size': cohort_size,
                'Metric': metric
            }
            
            for month in month_columns:
                row_data[month] = 0
            
            row_data['Total'] = 0
            table_data.append(row_data)
    
    filled_table = pd.DataFrame(table_data)
    
    # Convert month columns to float
    for month_col in month_columns:
        filled_table[month_col] = filled_table[month_col].astype(float)
    filled_table['Total'] = filled_table['Total'].astype(float)
    
    # Calculate metrics
    for idx, row in filled_table.iterrows():
        pome_month = row['POME Month']
        cohort_size = row['Cohort Size']
        metric = row['Metric']
        
        if cohort_size == 0:
            continue
        
        for month_col in month_columns:
            if month_col < pome_month:
                filled_table.at[idx, month_col] = 0
            else:
                filtered_data = raw_data[
                    (raw_data['month'] == month_col) & 
                    (raw_data['pome_month'] == pome_month) & 
                    (raw_data['IsFiltered'] == 1)
                ]
                
                if metric == 'Active Customers':
                    value = filtered_data['users'].sum()
                    filled_table.at[idx, month_col] = int(round(value, 0))
                elif metric == 'Quantities':
                    value = filtered_data['quantity'].sum()
                    filled_table.at[idx, month_col] = int(round(value, 0))
                elif metric == 'Orders':
                    value = filtered_data['orders'].sum()
                    filled_table.at[idx, month_col] = int(round(value, 0))
                elif metric == 'Revenue':
                    value = filtered_data['sales'].sum()
                    filled_table.at[idx, month_col] = int(round(value, 0))
    
    # Handle Cumulative Revenue calculation
    for idx, row in filled_table.iterrows():
        if row['Metric'] == 'Cumulative Revenue':
            pome_month = row['POME Month']
            
            revenue_row_idx = None
            for rev_idx, rev_row in filled_table.iterrows():
                if (rev_row['POME Month'] == pome_month and rev_row['Metric'] == 'Revenue'):
                    revenue_row_idx = rev_idx
                    break
            
            if revenue_row_idx is None:
                continue
                
            for month_col in month_columns:
                if month_col < pome_month:
                    filled_table.at[idx, month_col] = 0
                else:
                    current_month_index = month_columns.index(month_col)
                    revenue_values = []
                    for future_month in month_columns[current_month_index:]:
                        revenue_values.append(filled_table.at[revenue_row_idx, future_month])
                    
                    cumulative_sum = sum(revenue_values)
                    filled_table.at[idx, month_col] = int(round(cumulative_sum, 0))
    
    # Handle Retention Rate calculation
    for idx, row in filled_table.iterrows():
        if row['Metric'] == 'Retention Rate':
            pome_month = row['POME Month']
            cohort_size = row['Cohort Size']
            
            for month_col in month_columns:
                if month_col < pome_month:
                    filled_table.at[idx, month_col] = 0
                elif month_col == pome_month:
                    filtered_data = raw_data[
                        (raw_data['month'] == month_col) & 
                        (raw_data['pome_month'] == pome_month) & 
                        (raw_data['IsFiltered'] == 1)
                    ]
                    
                    orders_sum = filtered_data['orders'].sum()
                    if cohort_size > 0:
                        retention_rate = (orders_sum / cohort_size) - 1
                        filled_table.at[idx, month_col] = round(retention_rate*100, 2)
                    else:
                        filled_table.at[idx, month_col] = 0
                else:
                    active_customers_row_idx = None
                    for ac_idx, ac_row in filled_table.iterrows():
                        if (ac_row['POME Month'] == pome_month and ac_row['Metric'] == 'Active Customers'):
                            active_customers_row_idx = ac_idx
                            break
                    
                    if active_customers_row_idx is not None and cohort_size > 0:
                        active_customers = filled_table.at[active_customers_row_idx, month_col]
                        retention_rate = active_customers / cohort_size
                        filled_table.at[idx, month_col] = round(retention_rate*100, 4)
                    else:
                        filled_table.at[idx, month_col] = 0
    
    # Handle LTV calculation
    for idx, row in filled_table.iterrows():
        if row['Metric'] == 'LTV':
            pome_month = row['POME Month']
            cohort_size = row['Cohort Size']
            
            for month_col in month_columns:
                if month_col < pome_month:
                    filled_table.at[idx, month_col] = 0
                else:
                    revenue_row_idx = None
                    for rev_idx, rev_row in filled_table.iterrows():
                        if (rev_row['POME Month'] == pome_month and rev_row['Metric'] == 'Revenue'):
                            revenue_row_idx = rev_idx
                            break
                    
                    if revenue_row_idx is not None and cohort_size > 0:
                        revenue = filled_table.at[revenue_row_idx, month_col]
                        ltv = revenue / cohort_size
                        filled_table.at[idx, month_col] = round(ltv, 2)
                    else:
                        filled_table.at[idx, month_col] = 0
    
    # Calculate Total column
    for idx, row in filled_table.iterrows():
        total_value = sum([row[month] for month in month_columns])
        filled_table.at[idx, 'Total'] = total_value
    
    return filled_table, filter_msg

def retention_calculation(RawData, filled_table):

    df_raw_data = RawData.copy()
    df_raw_data['month'] = pd.to_datetime(df_raw_data['month'])
    df_raw_data['pmonth_date'] = pd.to_datetime(df_raw_data['pome_month'])

    # Create a dummy df_mom_retention based on your screenshot
    mom_retention_data = {
        'Month': (list(filled_table['POME Month'])),
        'Cohort': (list(filled_table['Cohort Size']))
    }

    df_mom_retention = pd.DataFrame(mom_retention_data).drop_duplicates().reset_index(drop=True)
    df_mom_retention['Month'] = pd.to_datetime(df_mom_retention['Month'])

    def calculate_retention(cohort_month, retention_month, df_raw_data):
        """
        Calculates the MoM retention value based on the Excel formula:
        =IF(D$3<$B14,"",SUMIFS(RawData!$I:$I,RawData!$D:$D,$B14,RawData!$A:$A,D$3)/SUMIFS(RawData!$F:$F,RawData!$D:$D,$B14,RawData!$A:$A,D$3))
        
        This calculates Average Revenue Per User (ARPU) = sales/users

        Args:
            cohort_month (pd.Timestamp): The initial acquisition month of the cohort ($B14).
            retention_month (pd.Timestamp): The month for which retention is being calculated (D$3).
            raw_data_df (pd.DataFrame): The DataFrame containing the raw sales and users data.

        Returns:
            float or str: The retention value (sales/users) or an empty string if
                        retention_month is before cohort_month.
                        Returns 0 if denominator (users) is 0 to avoid division by zero.
        """

        # IF(D$3<$B14,"",...)
        if retention_month < cohort_month:
            return ""

        # Filter raw data for the specific cohort month and retention month
        # SUMIFS conditions: RawData!$D:$D=$B4 AND RawData!$A:$A=D$3
        filtered_data = df_raw_data[
            (df_raw_data['pmonth_date'] == cohort_month) &  # $D:$D=$B4 (pome_month)
            (df_raw_data['month'] == retention_month)       # $A:$A=D$3 (month)
        ]

        # SUMIFS(RawData!$I:$I,...) = sum of sales column
        sum_sales = filtered_data['sales'].sum()
        
        # SUMIFS(RawData!$F:$F,...) = sum of users column  
        sum_users = filtered_data['users'].sum()

        if sum_users == 0:
            return 0  # To handle #DIV/0! equivalent, return 0 
        else:
            return sum_sales / sum_users  # This is ARPU (Average Revenue Per User)

    # --- 3. Apply the function to populate the MoM Retention table ---

    # Get unique cohort months from your df_mom_retention
    cohort_months = sorted(df_mom_retention['Month'].unique())

    # Instead of using all months, use only the cohort analysis months (starting from 2024-07-01)
    # This matches the cohort table structure
    retention_columns_str = sorted(cohort_months)  # Use the same months as cohort analysis
    retention_columns = [pd.to_datetime(month) for month in retention_columns_str]  # Convert to datetime for comparison


    # print(f"Cohort months: {[str(d) for d in cohort_months]}")
    # print(f"Retention columns: {retention_columns_str}")

    retention_table = pd.DataFrame(index=cohort_months, columns=retention_columns)
    # retention_table
    for cohort_m in cohort_months:
        for retention_m in retention_columns_str:
            retention_table.loc[cohort_m, retention_m] = calculate_retention(
                cohort_m, retention_m, df_raw_data
            )

    retention_table = retention_table.fillna('')
    return retention_table

def retention_calculation_v2(raw_data, cohort_table):
    """
    Calculate cumulative LTV retention table from cohort analysis
    
    Args:
        raw_data (pd.DataFrame): RawData DataFrame (not used but kept for consistency)
        cohort_table (pd.DataFrame): Cohort analysis table with LTV data
        
    Returns:
        pd.DataFrame: Retention table with cumulative LTV values
    """
    
    # Extract LTV data from cohort table
    ltv_data = cohort_table[cohort_table['Metric'] == 'LTV'].copy()
    
    if ltv_data.empty:
        # If no LTV data, return empty table
        return pd.DataFrame()
    
    # Get cohort information
    cohort_info = cohort_table[cohort_table['Metric'] == 'LTV'][['POME Month', 'Cohort Size']].drop_duplicates()
    
    # Get all month columns (excluding POME Month, Cohort Size, Metric, Total)
    month_columns = [col for col in ltv_data.columns if col not in ['POME Month', 'Cohort Size', 'Metric', 'Total']]
    month_columns = sorted(month_columns)  # Ensure chronological order
    
    print(f"Creating cumulative LTV retention table...")
    print(f"Found {len(ltv_data)} cohorts and {len(month_columns)} months")
    print(f"Months: {month_columns}")
    
    # Create the retention matrix
    cohort_months = sorted(ltv_data['POME Month'].unique())
    retention_data = []
    
    for cohort_month in cohort_months:
        # Get LTV data for this cohort
        cohort_ltv = ltv_data[ltv_data['POME Month'] == cohort_month].iloc[0]
        
        # Create row data
        row_data = {'POME Month': cohort_month}
        
        # Add cumulative LTV values for each month
        cumulative_ltv = 0
        for month_col in month_columns:
            # Only include months >= cohort month for cumulative calculation
            if month_col >= cohort_month:
                month_ltv = cohort_ltv[month_col] if month_col in cohort_ltv else 0
                
                # Handle numeric conversion
                if isinstance(month_ltv, (int, float)) and month_ltv > 0:
                    cumulative_ltv += month_ltv
                    row_data[month_col] = round(cumulative_ltv, 2)
                else:
                    row_data[month_col] = cumulative_ltv if cumulative_ltv > 0 else ""
            else:
                # Months before cohort month should be empty
                row_data[month_col] = ""
        
        retention_data.append(row_data)
    
    # Create DataFrame
    retention_table = pd.DataFrame(retention_data)
    
    # Add cohort sizes
    retention_table = retention_table.merge(
        cohort_info[['POME Month', 'Cohort Size']], 
        on='POME Month', 
        how='left'
    )
    
    # Reorder columns: POME Month, Cohort Size, then month columns
    column_order = ['POME Month', 'Cohort Size'] + month_columns
    retention_table = retention_table[column_order]
    
    # Fill any remaining NaN values
    retention_table = retention_table.fillna('')
    
    print(f"✅ Cumulative LTV retention table created with shape: {retention_table.shape}")
    
    return retention_table

def calculate_user_breakdown(raw_data, raw_data_wo_sku, selected_merchant_sku=None):
    """
    Calculate old users vs new users breakdown based on merchant SKU selection
    
    Args:
        raw_data (pd.DataFrame): RawData DataFrame
        selected_merchant_sku (str): Selected merchant SKU or None for all
        
    Returns:
        pd.DataFrame: DataFrame with month, all_users, new_users, old_users
    """
    
    # Apply SKU filtering if specified
    if selected_merchant_sku and selected_merchant_sku != "All":
        filtered_data = raw_data[raw_data['merchant_sku'] == selected_merchant_sku].copy()
    else:
        filtered_data = raw_data_wo_sku.copy()
    
    # Calculate new users (where pome_month equals month - first-time buyers)
    new_users_data = filtered_data[filtered_data['pome_month'] == filtered_data['month']].copy()
    new_users = new_users_data.groupby(['month']).agg({'users': 'sum'}).reset_index().rename(columns={'users': 'new_users'})
    
    # Calculate all users for each month
    all_users = filtered_data.groupby(['month']).agg({'users': 'sum'}).reset_index().rename(columns={'users': 'all_users'})
    
    # Combine and calculate old users
    combined_users = pd.merge(all_users, new_users, on='month', how='left')
    combined_users['new_users'] = combined_users['new_users'].fillna(0)
    combined_users['old_users'] = combined_users['all_users'] - combined_users['new_users']
    
    # Sort by month
    combined_users['month'] = pd.to_datetime(combined_users['month'])
    combined_users = combined_users.sort_values('month').reset_index(drop=True)
    
    return combined_users

def create_user_breakdown_chart(user_breakdown_df, selected_sku):
    """
    Create a stacked bar chart showing old users vs new users
    
    Args:
        user_breakdown_df (pd.DataFrame): DataFrame with user breakdown data
        selected_sku (str): Selected merchant SKU for title
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    
    # Create stacked bar chart
    fig = go.Figure()
    
    # Add new users bar
    fig.add_trace(go.Bar(
        name='New Users',
        x=user_breakdown_df['month'].dt.strftime('%Y-%m'),
        y=user_breakdown_df['new_users'],
        marker_color='#2E86AB',  # Blue
        text=user_breakdown_df['new_users'],
        textposition='inside',
        texttemplate='%{text}',
        hovertemplate='<b>New Users</b><br>' +
                      'Month: %{x}<br>' +
                      'Count: %{y:,}<extra></extra>'
    ))
    
    # Add old users bar
    fig.add_trace(go.Bar(
        name='Returning Users',
        x=user_breakdown_df['month'].dt.strftime('%Y-%m'),
        y=user_breakdown_df['old_users'],
        marker_color='#A23B72',  # Purple/Pink
        text=user_breakdown_df['old_users'],
        textposition='inside',
        texttemplate='%{text}',
        hovertemplate='<b>Returning Users</b><br>' +
                      'Month: %{x}<br>' +
                      'Count: %{y:,}<extra></extra>'
    ))
    
    # Update layout for stacked bar chart
    sku_text = f" - {selected_sku}" if selected_sku != "All" else " - All SKUs"
    
    fig.update_layout(
        title=f'User Breakdown by Month{sku_text}',
        xaxis_title='Month',
        yaxis_title='Number of Users',
        barmode='stack',
        hovermode='x unified',
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(t=80, b=40, l=40, r=40),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # Style the axes
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)',
        tickangle=45
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(128,128,128,0.2)'
    )
    
    return fig

def create_download_link(df, filename, file_label):
    """Create a download link for a dataframe"""
    csv = df.to_csv(index=False)
    b64 = io.StringIO(csv).getvalue().encode()
    href = f'<a href="data:file/csv;base64,{b64.hex()}" download="{filename}" style="text-decoration: none;"><button style="background-color: #4CAF50; color: white; padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer;">{file_label}</button></a>'
    return href

# Main Streamlit App
def main():
    st.title("📊 LTV Calculation Dashboard")
    st.markdown("Upload your CSV files and generate ProductRaw, ProductSummary, RawData, and Cohort Analysis")
    
    # Sidebar for file upload
    st.sidebar.header("📁 File Upload")
    uploaded_files = st.sidebar.file_uploader(
        "Choose CSV files",
        type="csv",
        accept_multiple_files=True,
        help="Upload multiple CSV files to combine and analyze"
    )
    
    if uploaded_files:
        st.sidebar.success(f"✅ {len(uploaded_files)} files uploaded")
        
        # Process files
        with st.spinner("Processing uploaded files..."):
            combined_df = process_files(uploaded_files)
        
        if combined_df is not None:
            st.success("Files processed successfully!")
            
            # Display basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", f"{len(combined_df):,}")
            with col2:
                st.metric("Total Columns", len(combined_df.columns))
            with col3:
                st.metric("Files Processed", len(uploaded_files))
            
            # Calculate all dataframes
            st.header("📈 Calculations")
            
            with st.spinner("Calculating ProductRaw..."):
                product_raw = calculate_product_raw(combined_df)
            
            with st.spinner("Calculating ProductSummary..."):
                product_summary = calculate_product_summary(combined_df)
            
            with st.spinner("Calculating RawData..."):
                raw_data = calculate_raw_data(combined_df)

            with st.spinner("Calculating RawDataWithoutSku..."):
                raw_data_wo_sku = calculate_raw_data_wo_sku(combined_df)
            
            st.success("All calculations completed!")
            
            # Display results in tabs
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["ProductRaw", "ProductSummary", "RawData", "Cohort Analysis", "Retention & LTV Analysis", "User Breakdown Analysis"])
            
            with tab1:
                st.subheader("ProductRaw Data")
                st.dataframe(product_raw, use_container_width=True)
                st.markdown(create_download_link(product_raw, "ProductRaw.csv", "📥 Download ProductRaw"), unsafe_allow_html=True)
            
            with tab2:
                st.subheader("ProductSummary Data")
                st.dataframe(product_summary, use_container_width=True)
                st.markdown(create_download_link(product_summary, "ProductSummary.csv", "📥 Download ProductSummary"), unsafe_allow_html=True)
            
            with tab3:
                st.subheader("RawData")
                st.dataframe(raw_data, use_container_width=True)
                st.markdown(create_download_link(raw_data, "RawData.csv", "📥 Download RawData"), unsafe_allow_html=True)
            
            with tab4:
                st.subheader("Cohort Analysis")
                
                # Merchant SKU selection
                available_skus = ["All"] + sorted(raw_data['merchant_sku'].unique().tolist())
                selected_sku = st.selectbox(
                    "Select Merchant SKU for Cohort Analysis:",
                    available_skus,
                    help="Choose a specific SKU or 'All' to analyze all SKUs together"
                )
                
                # Calculate cohort analysis
                with st.spinner(f"Calculating cohort analysis for {selected_sku}..."):
                    cohort_table, filter_msg = calculate_cohort_analysis(raw_data, selected_sku)
                
                st.info(filter_msg)
                
                # Display cohort table with scrolling
                st.subheader("Cohort Analysis Results")
                st.dataframe(
                    cohort_table, 
                    use_container_width=True,
                    height=600  # Make it scrollable
                )
                
                # Download button for cohort analysis
                st.markdown(create_download_link(cohort_table, f"Cohort_Analysis_{selected_sku}.csv", "📥 Download Cohort Analysis"), unsafe_allow_html=True)
            
            with tab5:
                st.header("📈 Retention & LTV Analysis")
                st.markdown("""
                This tab contains complementary analyses:
                 **Cumulative LTV Analysis** - Cumulative lifetime value from cohort data
                """)
                
                # # MoM Retention Analysis Section
                # st.subheader("📊 MoM Retention Analysis (ARPU)")
                # st.markdown("""
                # **Month-over-Month Retention Analysis** shows the Average Revenue Per User (ARPU) for each cohort across different months.
                
                # This analysis uses the Excel formula: 
                # `=IF(D$3<$B14,"",SUMIFS(sales,pome_month,cohort_month,month,analysis_month)/SUMIFS(users,pome_month,cohort_month,month,analysis_month))`
                # """)
                
                # # Calculate retention analysis
                # with st.spinner("Calculating MoM Retention..."):
                #     try:
                #         retention_table = retention_calculation(raw_data, cohort_table)
                        
                #         if retention_table is not None and not retention_table.empty:
                #             st.success("✅ MoM Retention calculated successfully!")
                            
                #             # # Display key metrics
                #             # col1, col2, col3 = st.columns(3)
                #             # with col1:
                #             #     total_cohorts = len(retention_table)
                #             #     st.metric("Total Cohorts", total_cohorts)
                #             # with col2:
                #             #     avg_cohort_size = retention_table['Cohort Size'].mean()
                #             #     st.metric("Avg Cohort Size", f"{avg_cohort_size:.0f}")
                #             # with col3:
                #             #     total_customers = retention_table['Cohort Size'].sum()
                #             #     st.metric("Total Customers", f"{total_customers:,}")
                            
                #             # Display retention table
                #             st.subheader("MoM Retention Table (ARPU by Cohort)")
                #             st.info("💡 Values represent Average Revenue Per User (ARPU) for each cohort in each month")
                            
                #             # Format the display table for better readability
                #             display_table = retention_table.copy()
                            
                #             # Format numeric columns (skip POME Month and Cohort Size)
                #             numeric_cols = [col for col in display_table.columns if col not in ['POME Month']]
                #             for col in numeric_cols:
                #                 display_table[col] = display_table[col].apply(
                #                     lambda x: f"${x:.2f}" if isinstance(x, (int, float)) and x != 0 else (x if x != 0 else "")
                #                 )
                            
                #             # Display with styling
                #             st.dataframe(
                #                 display_table,
                #                 use_container_width=True,
                #                 height=600
                #             )
                            
                #             # Download button
                #             st.markdown(create_download_link(retention_table, "MoM_Retention_Analysis.csv", "📥 Download MoM Retention"), unsafe_allow_html=True)
                            
                #             # Analysis insights
                #             st.subheader("📊 Key Insights")
                            
                #             # Calculate some insights
                #             numeric_retention_data = retention_table.copy()
                #             for col in numeric_cols:
                #                 numeric_retention_data[col] = pd.to_numeric(numeric_retention_data[col], errors='coerce')
                            
                #             # Find highest ARPU
                #             arpu_values = []
                #             for col in numeric_cols:
                #                 col_values = numeric_retention_data[col].dropna()
                #                 if len(col_values) > 0:
                #                     arpu_values.extend(col_values.tolist())
                            
                #             if arpu_values:
                #                 max_arpu = max([x for x in arpu_values if x > 0])
                #                 avg_arpu = np.mean([x for x in arpu_values if x > 0])
                                
                #                 col1, col2 = st.columns(2)
                #                 with col1:
                #                     st.metric("Highest ARPU", f"${max_arpu:.2f}")
                #                 with col2:
                #                     st.metric("Average ARPU", f"${avg_arpu:.2f}")
                #         else:
                #             st.error("❌ Failed to calculate retention analysis. Please check your data.")
                            
                #     except Exception as e:
                #         st.error(f"❌ Error calculating retention: {str(e)}")
                #         st.error("Please ensure your cohort analysis was calculated successfully first.")
                
                # # Add separator
                # st.markdown("---")
                
                # Cumulative LTV Analysis Section
                st.subheader("💰 Cumulative LTV Analysis")
                st.markdown("""
                **Cumulative Lifetime Value Analysis** shows the cumulative LTV for each cohort across different months. 
                This is calculated by taking the LTV values from the cohort analysis and showing them cumulatively from lowest month to highest month.
                
                💡 **How it works**: For each cohort, the LTV values are summed cumulatively across months, showing the total lifetime value accumulated over time.
                """)
                
                # Calculate cumulative LTV analysis
                with st.spinner("Calculating Cumulative LTV Analysis..."):
                    try:
                        cumulative_ltv_table = retention_calculation_v2(raw_data, cohort_table)
                        
                        if cumulative_ltv_table is not None and not cumulative_ltv_table.empty:
                            st.success("✅ Cumulative LTV Analysis calculated successfully!")
                            
                            # Display metrics for cumulative LTV
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                total_cohorts_ltv = len(cumulative_ltv_table)
                                st.metric("Total Cohorts", total_cohorts_ltv)
                            with col2:
                                # Find the highest cumulative LTV value
                                numeric_cols_ltv = [col for col in cumulative_ltv_table.columns if col not in ['POME Month', 'Cohort Size']]
                                max_ltv_values = []
                                for _, row in cumulative_ltv_table.iterrows():
                                    row_values = [row[col] for col in numeric_cols_ltv if isinstance(row[col], (int, float)) and row[col] > 0]
                                    if row_values:
                                        max_ltv_values.append(max(row_values))
                                
                                if max_ltv_values:
                                    highest_ltv = max(max_ltv_values)
                                    st.metric("Highest Cumulative LTV", f"${highest_ltv:.2f}")
                                else:
                                    st.metric("Highest Cumulative LTV", "$0.00")
                            with col3:
                                # Calculate average final LTV (last non-empty value for each cohort)
                                final_ltv_values = []
                                for _, row in cumulative_ltv_table.iterrows():
                                    row_values = [row[col] for col in reversed(numeric_cols_ltv) if isinstance(row[col], (int, float)) and row[col] > 0]
                                    if row_values:
                                        final_ltv_values.append(row_values[0])  # First non-zero from the right (latest month)
                                
                                if final_ltv_values:
                                    avg_final_ltv = sum(final_ltv_values) / len(final_ltv_values)
                                    st.metric("Avg Final LTV", f"${avg_final_ltv:.2f}")
                                else:
                                    st.metric("Avg Final LTV", "$0.00")
                            
                            # Display cumulative LTV table
                            st.subheader("Cumulative LTV Table by Cohort")
                            st.info("💡 Values represent cumulative LTV for each cohort across months. Values increase from left to right showing total lifetime value accumulation.")
                            
                            # Format the display table for better readability
                            display_table_ltv = cumulative_ltv_table.copy()
                            
                            # Format numeric columns (skip POME Month and Cohort Size)
                            numeric_cols_ltv = [col for col in display_table_ltv.columns if col not in ['POME Month', 'Cohort Size']]
                            for col in numeric_cols_ltv:
                                display_table_ltv[col] = display_table_ltv[col].apply(
                                    lambda x: f"${x:.2f}" if isinstance(x, (int, float)) and x != 0 else (x if x != 0 else "")
                                )
                            
                            # Display with styling
                            st.dataframe(
                                display_table_ltv,
                                use_container_width=True,
                                height=600
                            )
                            
                            # Download button for cumulative LTV
                            st.markdown(create_download_link(cumulative_ltv_table, "Cumulative_LTV_Analysis.csv", "📥 Download Cumulative LTV"), unsafe_allow_html=True)
                            
                        else:
                            st.error("❌ Failed to calculate cumulative LTV analysis. Please ensure your cohort analysis contains LTV data.")
                            
                    except Exception as e:
                        st.error(f"❌ Error calculating cumulative LTV: {str(e)}")
                        st.error("Please ensure your cohort analysis was calculated successfully and contains LTV metrics.")
            
            with tab6:
                st.header("👥 User Breakdown Analysis")
                st.markdown("""
                **New vs Returning Users Analysis** provides insights into customer acquisition and retention patterns by showing the breakdown of users each month.
                
                This analysis helps answer key business questions:
                - How many new customers are we acquiring each month?
                - What's the ratio of new vs returning customers?
                - How does user composition change over time?
                """)
                
                # Merchant SKU selection for user breakdown
                st.subheader("🎯 Analysis Configuration")
                available_skus = ["All"] + sorted(raw_data['merchant_sku'].unique().tolist())
                selected_sku_breakdown = st.selectbox(
                    "Select Merchant SKU for User Breakdown:",
                    available_skus,
                    help="Choose a specific SKU or 'All' to analyze user patterns across all SKUs",
                    key="user_breakdown_sku_selector"
                )
                
                st.markdown("---")
                
                # User breakdown explanation
                st.subheader("📖 Methodology")
                st.markdown("""
                **Calculation Logic**:
                - **New Users**: Users where their first purchase month (POME) equals the analysis month
                - **Returning Users**: Total users minus new users for each month
                - **Total Users**: All unique users who made purchases in each month
                
                This segmentation helps identify customer lifecycle patterns and acquisition effectiveness.
                """)
                
                # Calculate user breakdown
                with st.spinner(f"Calculating user breakdown for {selected_sku_breakdown}..."):
                    user_breakdown = calculate_user_breakdown(raw_data, raw_data_wo_sku, selected_sku_breakdown)
                
                if not user_breakdown.empty:
                    # Display key metrics
                    st.subheader("📊 Key Metrics")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        total_new_users = user_breakdown['new_users'].sum()
                        st.metric("Total New Users", f"{total_new_users:,}")
                    with col2:
                        total_returning_users = user_breakdown['old_users'].sum()
                        st.metric("Total Returning Users", f"{total_returning_users:,}")
                    with col3:
                        total_all_users = user_breakdown['all_users'].sum()
                        st.metric("Total Users", f"{total_all_users:,}")
                    with col4:
                        if total_all_users > 0:
                            new_user_percentage = (total_new_users / total_all_users) * 100
                            st.metric("New User %", f"{new_user_percentage:.1f}%")
                        else:
                            st.metric("New User %", "0%")
                    
                    # Additional insights
                    col1, col2 = st.columns(2)
                    with col1:
                        avg_new_users_per_month = user_breakdown['new_users'].mean()
                        st.metric("Avg New Users/Month", f"{avg_new_users_per_month:.0f}")
                    with col2:
                        avg_returning_users_per_month = user_breakdown['old_users'].mean()
                        st.metric("Avg Returning Users/Month", f"{avg_returning_users_per_month:.0f}")
                    
                    # Create and display the stacked bar chart
                    st.subheader("📈 User Breakdown Visualization")
                    fig = create_user_breakdown_chart(user_breakdown, selected_sku_breakdown)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Additional analysis insights
                    st.subheader("🔍 Analysis Insights")
                    
                    # Calculate trends
                    if len(user_breakdown) >= 2:
                        recent_new_users = user_breakdown.tail(3)['new_users'].mean()
                        earlier_new_users = user_breakdown.head(len(user_breakdown)-3)['new_users'].mean() if len(user_breakdown) > 3 else user_breakdown.head(1)['new_users'].iloc[0]
                        
                        if earlier_new_users > 0:
                            new_user_trend = ((recent_new_users - earlier_new_users) / earlier_new_users) * 100
                            trend_direction = "📈 Increasing" if new_user_trend > 5 else "📉 Decreasing" if new_user_trend < -5 else "➡️ Stable"
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.info(f"**New User Acquisition Trend**: {trend_direction} ({new_user_trend:+.1f}%)")
                            with col2:
                                best_month = user_breakdown.loc[user_breakdown['new_users'].idxmax()]
                                st.info(f"**Best Acquisition Month**: {best_month['month'].strftime('%Y-%m')} ({best_month['new_users']:.0f} new users)")
                    
                    # Display data table
                    st.subheader("📋 Detailed User Breakdown Data")
                    display_breakdown = user_breakdown.copy()
                    display_breakdown['month'] = display_breakdown['month'].dt.strftime('%Y-%m-%d')
                    display_breakdown = display_breakdown.rename(columns={
                        'month': 'Month',
                        'all_users': 'Total Users',
                        'new_users': 'New Users',
                        'old_users': 'Returning Users'
                    })
                    
                    # Add percentage columns
                    display_breakdown['New User %'] = ((display_breakdown['New Users'] / display_breakdown['Total Users']) * 100).round(1)
                    display_breakdown['Returning User %'] = ((display_breakdown['Returning Users'] / display_breakdown['Total Users']) * 100).round(1)
                    
                    st.dataframe(
                        display_breakdown,
                        use_container_width=True,
                        height=400
                    )
                    
                    # Download button for user breakdown
                    st.markdown(create_download_link(display_breakdown, f"User_Breakdown_{selected_sku_breakdown}.csv", "📥 Download User Breakdown Analysis"), unsafe_allow_html=True)
                    
                    # Summary insights
                    st.subheader("💡 Summary Insights")
                    total_months = len(user_breakdown)
                    months_with_growth = len(user_breakdown[user_breakdown['new_users'] > user_breakdown['old_users']])
                    
                    insights = [
                        f"📅 **Analysis Period**: {total_months} months of data",
                        f"🆕 **Acquisition-Heavy Months**: {months_with_growth} out of {total_months} months had more new users than returning users",
                        f"📊 **Customer Base Composition**: {new_user_percentage:.1f}% new users, {(100-new_user_percentage):.1f}% returning users"
                    ]
                    
                    for insight in insights:
                        st.markdown(insight)
                
                else:
                    st.warning("No user breakdown data available for the selected criteria.")
                    st.info("Please ensure you have processed data with valid purchase dates and user information.")
            
            # Download all results as ZIP
            st.header("📦 Download All Results")
            if st.button("📥 Download All as ZIP"):
                # Create ZIP file
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    # Add each DataFrame as CSV to ZIP
                    zip_file.writestr("ProductRaw.csv", product_raw.to_csv(index=False))
                    zip_file.writestr("ProductSummary.csv", product_summary.to_csv(index=False))
                    zip_file.writestr("RawData.csv", raw_data.to_csv(index=False))
                    zip_file.writestr(f"Cohort_Analysis_{selected_sku}.csv", cohort_table.to_csv(index=False))
                    
                    # Add retention analysis if available
                    try:
                        retention_table = retention_calculation(raw_data, cohort_table)
                        if retention_table is not None and not retention_table.empty:
                            zip_file.writestr("MoM_Retention_Analysis.csv", retention_table.to_csv(index=False))
                    except:
                        pass  # Skip if retention calculation fails
                    
                    # Add cumulative LTV analysis if available
                    try:
                        cumulative_ltv_table = retention_calculation_v2(raw_data, cohort_table)
                        if cumulative_ltv_table is not None and not cumulative_ltv_table.empty:
                            zip_file.writestr("Cumulative_LTV_Analysis.csv", cumulative_ltv_table.to_csv(index=False))
                    except:
                        pass  # Skip if cumulative LTV calculation fails
                    
                    # Add user breakdown analysis if available
                    try:
                        # Use the "All" option for ZIP download to include comprehensive data
                        user_breakdown = calculate_user_breakdown(raw_data, "All")
                        if not user_breakdown.empty:
                            display_breakdown = user_breakdown.copy()
                            display_breakdown['month'] = display_breakdown['month'].dt.strftime('%Y-%m-%d')
                            display_breakdown = display_breakdown.rename(columns={
                                'month': 'Month',
                                'all_users': 'Total Users',
                                'new_users': 'New Users',
                                'old_users': 'Returning Users'
                            })
                            # Add percentage columns
                            display_breakdown['New User %'] = ((display_breakdown['New Users'] / display_breakdown['Total Users']) * 100).round(1)
                            display_breakdown['Returning User %'] = ((display_breakdown['Returning Users'] / display_breakdown['Total Users']) * 100).round(1)
                            zip_file.writestr("User_Breakdown_Analysis.csv", display_breakdown.to_csv(index=False))
                    except:
                        pass  # Skip if user breakdown calculation fails
                
                zip_buffer.seek(0)
                
                st.download_button(
                    label="📥 Download ZIP File",
                    data=zip_buffer.getvalue(),
                    file_name="LTV_Analysis_Results.zip",
                    mime="application/zip"
                )
        else:
            st.error("Failed to process uploaded files. Please check the file format and try again.")
    
    else:
        st.info("👆 Please upload CSV files using the sidebar to get started.")
        
        # Show sample data format
        st.header("📋 Expected Data Format")
        st.markdown("""
        Your CSV files should contain the following columns:
        - **purchase_date**: Date of purchase
        - **buyer_email**: Customer email
        - **merchant_sku**: Product SKU
        - **amazon_order_id**: Order ID
        - **item_price**: Price of item
        - **shipped_quantity**: Quantity shipped
        - **currency**: Currency (USD)
        - **title**: Product title
        - And other relevant e-commerce data columns...
        """)

if __name__ == "__main__":
    main() 