import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder
from st_aggrid import GridOptionsBuilder, AgGrid, DataReturnMode, GridUpdateMode
from dateutil.parser import parse
import json

# ----------------- Helper Functions ---------------------------

def load_file(uploaded_file):
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

def detect_dates(df):
    date_cols = []
    for col in df.columns:
        try:
            pd.to_datetime(df[col])
            date_cols.append(col)
        except Exception:
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

def cap_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = np.where(df[col] < lower, lower, np.where(df[col] > upper, upper, df[col]))
    return df

def encode_features(df, col, method):
    if method == 'Label Encoding':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    elif method == 'OneHot Encoding':
        df = pd.get_dummies(df, columns=[col])
    return df

def fix_dates(df, col, date_format=None):
    try:
        if date_format:
            df[col] = pd.to_datetime(df[col], format=date_format, errors="coerce")
        else:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    except Exception:
        st.warning(f"Could not convert {col} using format {date_format or 'auto'}")
    return df

# ------------- Streamlit App -----------------------------------

st.set_page_config(layout="wide")
st.title("🧹 Data Cleaning Playground (Full Features)")

# ---- 1. Upload ----
uploaded_files = st.file_uploader(
    "Upload file(s): CSV, Excel, or JSON",
    type=['csv', 'xlsx', 'xls', 'json'],
    accept_multiple_files=True,
)

if uploaded_files:
    dfs = []
    if len(uploaded_files) == 1:
        df = load_file(uploaded_files[0])
        dfs.append(df)
    else:
        for uploaded_file in uploaded_files:
            dfi = load_file(uploaded_file)
            dfs.append(dfi)
        if st.checkbox("Concatenate all files (must have same columns)"):
            df = pd.concat(dfs, ignore_index=True)
        else:
            df_selector = st.selectbox('Select dataset to work on:', [f.name for f in uploaded_files])
            idx = [f.name for f in uploaded_files].index(df_selector)
            df = dfs[idx]
    if df is not None and (('df' not in st.session_state) or "just_uploaded" not in st.session_state or st.session_state["just_uploaded"]):
        st.session_state['df'] = df.copy()
        st.session_state["just_uploaded"] = False
else:
    st.session_state["just_uploaded"] = True
    st.info("Please upload data")
    st.stop()

df = st.session_state['df']

# ---- 2. Dataset Overview ----
st.header("Data Overview")
st.write("Shape:", df.shape)
st.write("Columns:", list(df.columns))
st.write("Data types:")
st.dataframe(pd.DataFrame(df.dtypes, columns=["dtype"]).T)
if df.shape[0] > 0:
    st.write("head(20):")
    st.dataframe(df.head(20), use_container_width=True)
with st.expander("Describe Data"):
    st.dataframe(df.describe(include='all').T)
st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# ---- 3. Edit Data Table ----
st.header("View/Edit Data Table")
max_to_show = min(100, df.shape[0])
num_rows = st.number_input("Rows to preview (max 2000):", 1, min(2000, df.shape[0]), max_to_show)
editable = st.checkbox("Enable Inline Editing? (top rows only)", value=False)
if df.shape[0] > 0:
    gb = GridOptionsBuilder.from_dataframe(df.iloc[:num_rows])
    gb.configure_pagination(enabled=True)
    gb.configure_default_column(editable=editable)
    grid_options = gb.build()
    grid_resp = AgGrid(
        df.iloc[:num_rows],
        gridOptions=grid_options,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        update_mode=(GridUpdateMode.MANUAL if editable else GridUpdateMode.NO_UPDATE),
        fit_columns_on_grid_load=True,
        enable_enterprise_modules=False,
        height=(min(num_rows,20) + 4) * 35,
        key='agrid1'
    )
    if editable:
        edited_df_part = grid_resp['data']
        df_update_idx = edited_df_part.index
        for i in df_update_idx:
            st.session_state['df'].loc[i] = edited_df_part.loc[i]
        df = st.session_state['df']
    if st.button("Show all data table (caution, can be slow!)"):
        AgGrid(df, fit_columns_on_grid_load=True, height=min(500, df.shape[0]*30))

# ---- 4. Rename Columns ----
st.header("Rename/Delete Columns")
# Rename
if df.shape[1] > 0:
    col_to_rename = st.selectbox("Column to rename:", df.columns, key="rename_col")
    new_col_name = st.text_input("New name:", value=col_to_rename, key="rename_new")
    if st.button("Rename Column", key="rename_btn"):
        if new_col_name in df.columns:
            st.warning("A column with that name already exists.")
        elif not new_col_name.strip():
            st.warning("Column name cannot be empty.")
        else:
            df = df.rename(columns={col_to_rename: new_col_name})
            st.session_state['df'] = df
            st.success(f"Renamed column successfully.")
            st.experimental_rerun()
# Delete
cols_to_drop = st.multiselect("Select columns to delete:", df.columns, key="drop_cols")
if st.button("Delete Selected Column(s)", key="delete_btn"):
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        st.session_state['df'] = df
        st.success("Selected column(s) deleted successfully!")
        st.experimental_rerun()
    else:
        st.warning("No columns selected to delete.")

# ---- 5. Replace Values in Column ----
st.header("Replace Value in Column")
col_for_replace = st.selectbox("Column:", df.columns, key="replace_col")
unique_vals = df[col_for_replace].unique()
try:
    old_val = st.selectbox("Value to replace:", unique_vals, key="replace_old_val")
except Exception:  # If all unique values are nan
    old_val = None
new_val = st.text_input("Replacement value:", value=str(old_val), key="replace_new_val")
if st.button("Replace value globally", key="replace_btn"):
    if old_val is not None:
        df[col_for_replace] = df[col_for_replace].replace(old_val, new_val)
        st.session_state['df'] = df
        st.success(f"Replaced all '{old_val}' with '{new_val}' in {col_for_replace}")

# ---- 6. Validate Numeric Range ----
st.header("Range Validation")
num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
if num_cols:
    col_range = st.selectbox("Numeric column:", num_cols, key="range_col")
    if df[col_range].dropna().shape[0] > 0:
        col_min, col_max = float(df[col_range].min()), float(df[col_range].max())
        vmin = st.number_input("Min valid value:", value=col_min)
        vmax = st.number_input("Max valid value:", value=col_max)
        if st.button("Apply Range Validation", key="val_range_btn"):
            before = df.shape[0]
            df = df[(df[col_range] >= vmin) & (df[col_range] <= vmax)]
            st.session_state['df'] = df
            st.success(f"Kept {df.shape[0]}/{before} rows within requested range.")
            st.experimental_rerun()

# ---- 7. Duplicate Handling ----
st.header("Duplicate Handling & Removal")
d_count = df.duplicated().sum()
st.write(f"Number of duplicate rows: {d_count}")
if d_count > 0 and st.button("Remove duplicates (all columns)", key="dup_btn"):
    df = df.drop_duplicates()
    st.session_state['df'] = df
    st.success("Removed duplicate rows.")
    st.experimental_rerun()

# ---- 8. Missing Value Handling (Batch) ----
st.header("Missing Value Handling (Batch)")
nulls = df.isnull().sum()
st.write("Nulls per column:", nulls)
missing_cols = [col for col in df.columns if df[col].isnull().any()]
if missing_cols:
    batch_mv_cols = st.multiselect("Columns to fix NAs:", missing_cols, default=missing_cols, key="mv_cols")
    mv_choice = st.radio("Strategy:",
                         ["Delete rows (with any NA in these cols)",
                          "Fill mean",
                          "Fill mode",
                          "Fill median",
                          "KNN Impute (numeric only)",
                          "Fill constant"], key="mv_radio")
    const_val = None
    if mv_choice == "Fill constant":
        const_val = st.text_input("Constant value:", "missing", key="fillconst_val")
    k = None
    if mv_choice == "KNN Impute (numeric only)":
        k = st.number_input("Neighbors K:", 1, 20, 3, key="knn_k")
    if st.button("Apply Missing Value Handling", key="mv_btn"):
        temp_df = df.copy()
        if mv_choice == "Delete rows (with any NA in these cols)":
            temp_df = temp_df.dropna(subset=batch_mv_cols)
            st.success("Rows with missing values removed.")
        elif mv_choice == "Fill mean":
            for c in batch_mv_cols:
                if pd.api.types.is_numeric_dtype(temp_df[c]):
                    temp_df[c] = temp_df[c].fillna(temp_df[c].mean())
            st.success("Missing values filled with mean in selected columns.")
        elif mv_choice == "Fill mode":
            for c in batch_mv_cols:
                temp_df[c] = temp_df[c].fillna(temp_df[c].mode().iloc[0])
            st.success("Missing values filled with mode in selected columns.")
        elif mv_choice == "Fill median":
            for c in batch_mv_cols:
                if pd.api.types.is_numeric_dtype(temp_df[c]):
                    temp_df[c] = temp_df[c].fillna(temp_df[c].median())
            st.success("Missing values filled with median in selected columns.")
        elif mv_choice == "KNN Impute (numeric only)":
            knn_cols = [c for c in batch_mv_cols if pd.api.types.is_numeric_dtype(temp_df[c])]
            if knn_cols:
                imp = KNNImputer(n_neighbors=int(k))
                temp_df[knn_cols] = imp.fit_transform(temp_df[knn_cols])
                st.success("Missing values imputed via KNN.")
        elif mv_choice == "Fill constant":
            for c in batch_mv_cols:
                temp_df[c] = temp_df[c].fillna(const_val)
            st.success(f"Missing values filled with constant: {const_val}")
        st.session_state['df'] = temp_df
        st.experimental_rerun()
else:
    st.success("No missing values!")

# ---- 9. Date Validation ----
st.header("Date Validation/Conversion")
potential_dates = detect_dates(df)
if potential_dates:
    col_date = st.selectbox("Column to treat as date:", potential_dates, key="date_sel")
    st.write(f"Sample: {df[col_date].iloc[:5].tolist()}")
    date_format = st.text_input("Date format (optional, e.g. %Y-%m-%d):", value="", key="date_fmt")
    if st.button("Try date conversion", key="date_btn"):
        before = df[col_date].isna().sum()
        df = fix_dates(df, col_date, date_format if date_format else None)
        after = df[col_date].isna().sum()
        st.session_state['df'] = df
        st.success(f"Converted. NAs before: {before}, after: {after}")
        st.experimental_rerun()
else:
    st.info("No potential date columns detected.")

# ---- 10. Outlier Handling (Batch) ----
st.header("Outlier Detection & Handling")
if num_cols:
    outlier_info = {col: outlier_detection(df, col) for col in num_cols}
    st.write("Outliers per column:")
    for col in outlier_info:
        st.write(f"{col}: {len(outlier_info[col])}")
    total_outliers = sum(len(v) for v in outlier_info.values())
    st.write(f"Total outliers: {total_outliers}")
    batch_outlier_cols = st.multiselect("Numeric columns for outlier handling:", num_cols, default=num_cols, key="outlier_cols")
    out_action = st.radio("How to handle outliers in selected columns?", ["Keep", "Delete", "Cap"], key="outlier_radio")
    if st.button("Apply Outlier Handling", key="outlier_btn"):
        temp_df = df.copy()
        if batch_outlier_cols:
            if out_action == "Delete":
                idxs_to_del = []
                for col in batch_outlier_cols:
                    idxs_to_del.extend(outlier_info[col])
                idxs_to_del = list(set(idxs_to_del))
                before = temp_df.shape[0]
                temp_df = temp_df.drop(index=idxs_to_del)
                st.success(f"Deleted {before - temp_df.shape[0]} rows with outliers in selected columns.")
            elif out_action == "Cap":
                for col in batch_outlier_cols:
                    temp_df = cap_outliers(temp_df, col)
                st.success("Capped outliers in selected columns.")
            elif out_action == "Keep":
                st.info("Outliers kept, no changes made.")
            st.session_state['df'] = temp_df
            st.experimental_rerun()
        else:
            st.warning("Select at least one numeric column.")

# ---- 11. Normalization / Transformation ----
st.header("Normalization / Transformation")
if num_cols:
    norm_col = st.selectbox("Column to normalize:", num_cols, key="norm_col")
    n_method = st.selectbox("Scaler type:", ["MinMaxScaler", "StandardScaler", "RobustScaler"], key="norm_method")
    if st.button("Apply normalization", key="norm_btn"):
        scaler = {'MinMaxScaler': MinMaxScaler(), 'StandardScaler': StandardScaler(), 'RobustScaler': RobustScaler()}[n_method]
        df[[norm_col]] = scaler.fit_transform(df[[norm_col]])
        st.session_state['df'] = df
        st.success(f"{norm_col} normalized with {n_method}")
        st.experimental_rerun()

# ---- 12. Feature Engineering ----
st.header("Feature Engineering")
feature_eng_option = st.selectbox("Choose:", ['None', 'Polynomial Feature', 'Encode Categorical'], key="feateng_opt")
if feature_eng_option == "Polynomial Feature" and num_cols:
    col_poly = st.selectbox("Numeric column:", num_cols, key="poly_col")
    degree = st.slider("Degree:", 2, 5, 2)
    for d in range(2, degree+1):
        df[f"{col_poly}^{d}"] = df[col_poly] ** d
    st.session_state['df'] = df
    st.success(f"Added {col_poly} to the degree {degree}")
    st.experimental_rerun()
elif feature_eng_option == "Encode Categorical":
    cat_cols = [c for c in df.columns if df[c].dtype == "object"]
    if cat_cols:
        col_enc = st.selectbox("Categorical column:", cat_cols, key="catenc_col")
        enc_method = st.selectbox("Encoding:", ["Label Encoding", "OneHot Encoding"], key="catenc_method")
        df = encode_features(df, col_enc, enc_method)
        st.session_state['df'] = df
        st.success(f"{col_enc} encoded via {enc_method}.")
        st.experimental_rerun()
    else:
        st.info("No object/categorical columns found.")

# ---- 13. Download ----
st.header("Download Cleaned Data")
buf = StringIO()
df.to_csv(buf, index=False)
st.download_button("Download as CSV", buf.getvalue(), file_name="cleaned_data.csv")

st.info("All edits are immediately applied! Download for latest result.")
