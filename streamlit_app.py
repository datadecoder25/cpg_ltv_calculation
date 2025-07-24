import streamlit as st
import pandas as pd
import numpy as np
import io
import zipfile
from datetime import datetime
import math

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
            
            st.success("All calculations completed!")
            
            # Display results in tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["ProductRaw", "ProductSummary", "RawData", "Cohort Analysis", "MoM Retention"])
            
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
                
                # # Additional metrics display
                # if not cohort_table.empty:
                #     st.subheader("📊 Summary Metrics")
                    
                #     # Filter for specific metrics to show summary
                #     revenue_rows = cohort_table[cohort_table['Metric'] == 'Revenue']
                #     active_customers_rows = cohort_table[cohort_table['Metric'] == 'Active Customers']
                    
                #     if not revenue_rows.empty:
                #         total_revenue = revenue_rows['Total'].sum()
                #         st.metric("Total Revenue Across All Cohorts", f"${total_revenue:,.2f}")
                    
                #     if not active_customers_rows.empty:
                #         total_customers = active_customers_rows['Total'].sum()
                #         st.metric("Total Active Customer Interactions", f"{total_customers:,}")
            
            with tab5:
                st.subheader("📈 MoM Retention Analysis")
                st.markdown("""
                **Month-over-Month Retention Analysis** shows the Average Revenue Per User (ARPU) for each cohort across different months.
                
                This analysis uses the Excel formula: 
                `=IF(D$3<$B14,"",SUMIFS(sales,pome_month,cohort_month,month,analysis_month)/SUMIFS(users,pome_month,cohort_month,month,analysis_month))`
                """)
                
                # Calculate retention analysis
                with st.spinner("Calculating MoM Retention..."):
                    try:
                        retention_table = retention_calculation(raw_data, cohort_table)
                        
                        if retention_table is not None and not retention_table.empty:
                            st.success("✅ MoM Retention calculated successfully!")
                            
                            # # Display key metrics
                            # col1, col2, col3 = st.columns(3)
                            # with col1:
                            #     total_cohorts = len(retention_table)
                            #     st.metric("Total Cohorts", total_cohorts)
                            # with col2:
                            #     avg_cohort_size = retention_table['Cohort Size'].mean()
                            #     st.metric("Avg Cohort Size", f"{avg_cohort_size:.0f}")
                            # with col3:
                            #     total_customers = retention_table['Cohort Size'].sum()
                            #     st.metric("Total Customers", f"{total_customers:,}")
                            
                            # Display retention table
                            st.subheader("MoM Retention Table (ARPU by Cohort)")
                            st.info("💡 Values represent Average Revenue Per User (ARPU) for each cohort in each month")
                            
                            # Format the display table for better readability
                            display_table = retention_table.copy()
                            
                            # Format numeric columns (skip POME Month and Cohort Size)
                            numeric_cols = [col for col in display_table.columns if col not in ['POME Month']]
                            for col in numeric_cols:
                                display_table[col] = display_table[col].apply(
                                    lambda x: f"${x:.2f}" if isinstance(x, (int, float)) and x != 0 else (x if x != 0 else "")
                                )
                            
                            # Display with styling
                            st.dataframe(
                                display_table,
                                use_container_width=True,
                                height=600
                            )
                            
                            # Download button
                            st.markdown(create_download_link(retention_table, "MoM_Retention_Analysis.csv", "📥 Download MoM Retention"), unsafe_allow_html=True)
                            
                            # Analysis insights
                            st.subheader("📊 Key Insights")
                            
                            # Calculate some insights
                            numeric_retention_data = retention_table.copy()
                            for col in numeric_cols:
                                numeric_retention_data[col] = pd.to_numeric(numeric_retention_data[col], errors='coerce')
                            
                            # Find highest ARPU
                            arpu_values = []
                            for col in numeric_cols:
                                col_values = numeric_retention_data[col].dropna()
                                if len(col_values) > 0:
                                    arpu_values.extend(col_values.tolist())
                            
                            if arpu_values:
                                max_arpu = max([x for x in arpu_values if x > 0])
                                avg_arpu = np.mean([x for x in arpu_values if x > 0])
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Highest ARPU", f"${max_arpu:.2f}")
                                with col2:
                                    st.metric("Average ARPU", f"${avg_arpu:.2f}")
                        else:
                            st.error("❌ Failed to calculate retention analysis. Please check your data.")
                            
                    except Exception as e:
                        st.error(f"❌ Error calculating retention: {str(e)}")
                        st.error("Please ensure your cohort analysis was calculated successfully first.")
            
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