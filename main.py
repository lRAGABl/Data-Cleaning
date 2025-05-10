import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder
from st_aggrid import GridOptionsBuilder, AgGrid, DataReturnMode, GridUpdateMode, JsCode
from datetime import datetime
from dateutil.parser import parse
import json

# ============ Helper functions =============

def load_file(uploaded_file):
    if uploaded_file is not None:
        filename = uploaded_file.name
        if filename.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif filename.endswith('.xlsx') or filename.endswith('.xls'):
            return pd.read_excel(uploaded_file)
        elif filename.endswith('.json'):
            content = uploaded_file.read()
            decoded = content.decode('utf-8')
            json_data = json.loads(decoded)
            return pd.json_normalize(json_data)
        else:
            st.error('Unsupported file format.')
            return None
    return None

def detect_dates(df):
    date_cols = []
    for col in df.columns:
        try:
            pd.to_datetime(df[col])
            date_cols.append(col)
        except:
            pass
    return date_cols

def outlier_detection(df, col):
    if not np.issubdtype(df[col].dtype, np.number):
        return []
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)].index.tolist()
    return outliers

def replace_in_column(df, column, old_val, new_val):
    df[column] = df[column].replace(old_val, new_val)
    return df

def column_range_validation(df, col, min_val, max_val):
    df = df[(df[col] >= min_val) & (df[col] <= max_val)]
    return df

def rename_column(df, old, new):
    df = df.rename(columns={old: new})
    return df

def fix_dates(df, col, date_format):
    try:
        df[col] = pd.to_datetime(df[col], format=date_format, errors='coerce')
    except:
        st.warning(f"Could not convert {col} using format {date_format}")
    return df

def cap_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = np.where(df[col] < lower, lower,
                       np.where(df[col] > upper, upper, df[col]))
    return df

def encode_features(df, col, method):
    if method == 'Label Encoding':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    elif method == 'OneHot Encoding':
        df = pd.get_dummies(df, columns=[col])
    return df

# ============ Main Streamlit App =============

st.title('🧹 Advanced Data Cleaning Playground')

uploaded_files = st.file_uploader("Upload your file(s) (CSV, Excel, or JSON)", type=['csv', 'xlsx', 'xls', 'json'], accept_multiple_files=True)
if uploaded_files:
    dfs = []
    if len(uploaded_files) == 1:
        df = load_file(uploaded_files[0])
        dfs.append(df)
    else:
        for uploaded_file in uploaded_files:
            dfi = load_file(uploaded_file)
            dfs.append(dfi)
        # Option: merge/join files or keep separate
        if st.checkbox('Concatenate all files (vertically, same columns required)?'):
            df = pd.concat(dfs)
        else:
            df_selector = st.selectbox('Select dataset to work on:', [f.name for f in uploaded_files])
            idx = [f.name for f in uploaded_files].index(df_selector)
            df = dfs[idx]
    # Store df in session so edits persist
    if 'df' not in st.session_state or st.session_state['df'].equals(df) == False:
        st.session_state['df'] = df.copy()
else:
    st.info('Please upload your dataset(s).')
    st.stop()

df = st.session_state['df']

# --- Info ---
st.subheader('Dataset Overview')
st.write('**Shape:**', df.shape)
st.write('**Columns:**', list(df.columns))
st.write('**Data types:**')
st.write(df.dtypes.astype(str))
st.write('**head(20):**')
st.write(df.head(20))
with st.expander('Describe Data'):
    st.write(df.describe(include='all').T)
st.write(f"**Number of Rows:** {df.shape[0]}")
st.write(f"**Number of Columns:** {df.shape[1]}")

# --- Show Editable Table ---
st.subheader('Editable Data Table')
num_rows = st.slider('Number of rows to show:', 10, min(5000, df.shape[0]), 100)
editable = st.checkbox('Enable Inline Editing?', value=True)
# Use AgGrid for better interactive editing and features
gb = GridOptionsBuilder.from_dataframe(df.iloc[:num_rows])
gb.configure_pagination(enabled=True)
gb.configure_default_column(editable=editable)
gb.configure_side_bar()
grid_options = gb.build()

grid_resp = AgGrid(df.iloc[:num_rows],
                   gridOptions=grid_options,
                   data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                   update_mode=GridUpdateMode.MANUAL if editable else GridUpdateMode.NO_UPDATE,
                   fit_columns_on_grid_load=True,
                   enable_enterprise_modules=False,
                   height=(min(num_rows,20) + 2) * 35,
                   key='agrid1'
                  )
edited_df_part = grid_resp['data']

if editable:
    # Update changes in session df (for the visible part)
    df_update_idx = edited_df_part.index
    for i in df_update_idx:
        st.session_state['df'].loc[i] = edited_df_part.loc[i]
    df = st.session_state['df']

if st.button('Show all data (may be slow for big tables)'):
    AgGrid(df, fit_columns_on_grid_load=True, height=min(500, df.shape[0]*30))

# --- Rename columns ---
st.subheader('Column Operations')
col_to_rename = st.selectbox('Choose column to rename:', df.columns)
new_col_name = st.text_input('New name:', value=col_to_rename)
if st.button('Rename Column!'):
    df = rename_column(df, col_to_rename, new_col_name)
    st.session_state['df'] = df
    st.success(f"Renamed {col_to_rename} to {new_col_name}")
    st.experimental_rerun()

# --- Replace values globally in a column ---
st.markdown("**Replace all occurrences of a value in a column**")
col_for_replace = st.selectbox("Column for value replacement:", df.columns)
unique_vals = df[col_for_replace].unique()
old_val = st.selectbox('Old value to replace:', unique_vals)
new_val = st.text_input('Replacement:', value=str(old_val))
if st.button('Replace value globally'):
    df = replace_in_column(df, col_for_replace, old_val, new_val)
    st.session_state['df'] = df
    st.success(f"Replaced all {old_val} in {col_for_replace} with {new_val}")

# --- Column Range Validation ---
st.subheader('Validate Numeric Ranges')
num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
if num_cols:
    col_to_validate = st.selectbox('Select numeric column for range validation:', num_cols)
    col_min, col_max = float(df[col_to_validate].min()), float(df[col_to_validate].max())
    vmin = st.number_input('Min valid value:', value=col_min)
    vmax = st.number_input('Max valid value:', value=col_max)
    if st.button('Apply Range Validation'):
        before = df.shape[0]
        df = column_range_validation(df, col_to_validate, vmin, vmax)
        after = df.shape[0]
        st.session_state['df'] = df
        st.success(f"Filtered out {before-after} rows outside ({vmin}, {vmax})")

# --- Duplicate Handling ---
st.subheader('Duplicate Detection')
d_count = df.duplicated().sum()
st.write(f"Number of duplicate rows: {d_count}")
if d_count > 0 and st.button('Remove all duplicate rows'):
    df = df.drop_duplicates()
    st.session_state['df'] = df
    st.success('Removed duplicate rows!')

# --- Missing Value Handling (Batch) ---
st.subheader('Missing Value Analysis and Handling')
null_stats = df.isnull().sum()
st.write(f"Null counts per column: {null_stats}")
null_rows = df.isnull().any(axis=1).sum()
st.write(f"Rows with at least one missing value: {null_rows}")

missing_cols = [col for col in df.columns if df[col].isnull().any()]
if missing_cols:
    batch_mv_cols = st.multiselect("Select columns to handle missing values (batch):", missing_cols, default=missing_cols)
    mv_choice = st.radio("Strategy for these columns:", [
        "Delete rows with missing (any selected column)",
        "Fill with mean (numeric)",
        "Fill with mode (numeric or categorical)",
        "Fill with median (numeric)",
        "KNN Imputer (numeric only)",
        "Fill with constant"
    ])
    if mv_choice == "Fill with constant":
        const_val = st.text_input('Constant value to fill:', value="0")
    if st.button('Apply Missing Value Handling to Selected Columns'):
        if mv_choice == "Delete rows with missing (any selected column)":
            before = df.shape[0]
            df = df.dropna(subset=batch_mv_cols)
            st.success(f"Removed {before - df.shape[0]} rows with missing in selected columns!")
        elif mv_choice == "Fill with mean (numeric)":
            for c in batch_mv_cols:
                if pd.api.types.is_numeric_dtype(df[c]):
                    df[c] = df[c].fillna(df[c].mean())
            st.success("Filled NAs with mean for selected columns.")
        elif mv_choice == "Fill with mode (numeric or categorical)":
            for c in batch_mv_cols:
                df[c] = df[c].fillna(df[c].mode()[0])
            st.success("Filled NAs with mode for selected columns.")
        elif mv_choice == "Fill with median (numeric)":
            for c in batch_mv_cols:
                if pd.api.types.is_numeric_dtype(df[c]):
                    df[c] = df[c].fillna(df[c].median())
            st.success("Filled NAs with median for selected columns.")
        elif mv_choice == "KNN Imputer (numeric only)":
            numeric_batch = [c for c in batch_mv_cols if pd.api.types.is_numeric_dtype(df[c])]
            if numeric_batch:
                k = st.number_input('Set K for KNN:', 1, 20, 3)
                imputer = KNNImputer(n_neighbors=int(k))
                df[numeric_batch] = imputer.fit_transform(df[numeric_batch])
                st.success(f"KNN-imputed selected numeric columns (k={k}).")
            else:
                st.warning("No numeric columns selected for KNN.")
        elif mv_choice == "Fill with constant":
            for c in batch_mv_cols:
                df[c] = df[c].fillna(const_val)
            st.success(f"Filled NAs with constant '{const_val}' for selected columns.")
        st.session_state['df'] = df
else:
    st.success("No columns have missing values.")

# --- Date Validation ---
st.subheader('Date Handling / Validation')
potential_dates = detect_dates(df)
if potential_dates:
    col_date = st.selectbox('Pick column to validate as date:', potential_dates)
    st.write(f"Sample values: {df[col_date].iloc[:5].tolist()}")
    date_format = st.text_input('Date format (optional, e.g., %Y-%m-%d):', value='')
    if st.button('Try date conversion'):
        before = df[col_date].isna().sum()
        df = fix_dates(df, col_date, date_format if date_format else None)
        after = df[col_date].isna().sum()
        st.session_state['df'] = df
        st.success(f"Converted. Nulls before: {before}, Nulls after: {after}")
else:
    st.info('No obvious date columns detected.')

# --- Outlier Detection and Handling (Batch) ---
st.subheader('Outlier Detection & Handling')
if num_cols:
    outlier_info = {col: outlier_detection(df, col) for col in num_cols}
    st.write('Outliers per column:')
    for col in outlier_info:
        st.write(f"{col}: {len(outlier_info[col])}")
    total_outliers = sum(len(v) for v in outlier_info.values())
    st.write(f"Total outliers: {total_outliers}")

    batch_outlier_cols = st.multiselect("Select numeric columns to handle outliers (batch):", num_cols)
    out_action = st.radio('How to handle outliers in selected columns?', ['Keep', 'Delete', 'Cap'])
    if st.button('Apply Outlier Handling to Selected Columns'):
        if not batch_outlier_cols:
            st.warning("Please select at least one column.")
        elif out_action == 'Delete':
            # Delete any row with outlier in any selected col
            idxs_to_del = []
            for col in batch_outlier_cols:
                idxs_to_del.extend(outlier_info[col])
            idxs_to_del = list(set(idxs_to_del))
            before = df.shape[0]
            df = df.drop(index=idxs_to_del)
            st.success(f"Deleted {before - df.shape[0]} rows with outliers in selected columns.")
        elif out_action == 'Cap':
            for col in batch_outlier_cols:
                df = cap_outliers(df, col)
            st.success(f"Capped outliers in selected columns.")
        elif out_action == 'Keep':
            st.info("No changes made. Outliers kept.")
        st.session_state['df'] = df

# --- Normalization / Transformation ---
st.subheader('Data Normalization/Transformer')
if num_cols:
    norm_col = st.selectbox('Select column to normalize:', num_cols)
    n_method = st.selectbox('Scaler type:', ['MinMaxScaler', 'StandardScaler', 'RobustScaler'])
    if st.button('Apply normalization'):
        scaler = {'MinMaxScaler': MinMaxScaler(), 'StandardScaler': StandardScaler(), 'RobustScaler': RobustScaler()}[n_method]
        df[[norm_col]] = scaler.fit_transform(df[[norm_col]])
        st.session_state['df'] = df
        st.success(f"{norm_col} normalized with {n_method}")

# --- Feature Engineering (simple demo) ---
st.subheader('Feature Engineering')
option = st.selectbox('Choose:', ['None', 'Polynomial Feature', 'Encode Categorical'])
if option == 'Polynomial Feature' and num_cols:
    col_poly = st.selectbox('Select numeric column:', num_cols)
    degree = st.slider('Degree:', 2, 5, 2)
    for d in range(2, degree+1):
        df[f'{col_poly}^{d}'] = df[col_poly] ** d
    st.session_state['df'] = df
    st.success(f"Added degrees up to {degree} for {col_poly}")
elif option == 'Encode Categorical':
    cat_cols = [c for c in df.columns if df[c].dtype == 'object']
    if cat_cols:
        col_enc = st.selectbox('Select categorical column:', cat_cols)
        enc_method = st.selectbox('Encoding:', ['Label Encoding', 'OneHot Encoding'])
        df = encode_features(df, col_enc, enc_method)
        st.session_state['df'] = df
        st.success(f"{col_enc} encoded by {enc_method}")
    else:
        st.info("No object-type columns available for encoding.")

# --- End ---
with st.expander('Download Cleaned Dataframe'):
    buf = StringIO()
    df.to_csv(buf, index=False)
    st.download_button("Download as CSV", data=buf.getvalue(), file_name="cleaned_data.csv", mime="text/csv")

st.info('All edits are live on the dataframe in session. Re-download to capture the latest changes.')
