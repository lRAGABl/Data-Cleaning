import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats
import io

st.set_page_config(layout="wide")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None

def load_data(uploaded_files):
    dfs = []
    for file in uploaded_files:
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.name.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file)
            elif file.name.endswith('.json'):
                df = pd.read_json(file)
            dfs.append(df)
        except Exception as e:
            st.error(f"Error loading {file.name}: {str(e)}")
    return pd.concat(dfs) if len(dfs) > 1 else dfs[0] if dfs else None

uploaded_files = st.sidebar.file_uploader("Upload datasets", accept_multiple_files=True, type=['csv', 'xlsx', 'json'])

if uploaded_files:
    if st.session_state.df is None:
        st.session_state.df = load_data(uploaded_files)
    
    if st.session_state.df is not None:
        df = st.session_state.df
        
        st.header("Data Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Info")
            buffer = io.StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())
            
        with col2:
            st.subheader("Basic Statistics")
            st.write(df.describe())
        
        st.subheader("First 20 Rows")
        st.dataframe(df.head(20))
        
        # Editable Data Table
        st.header("Interactive Data Editor")
        num_rows = st.slider("Select number of rows to display", 100, len(df), 100)
        edited_df = st.data_editor(df.head(num_rows), num_rows="fixed", use_container_width=True)
        df.update(edited_df)
        st.session_state.df = df  # Update session state

        # Column Management
        st.header("Column Management")
        selected_col = st.selectbox("Select column to rename", df.columns)
        new_name = st.text_input("New column name", selected_col)
        if st.button("Rename Column"):
            df.rename(columns={selected_col: new_name}, inplace=True)
            st.session_state.df = df
            st.rerun()

        st.subheader("Data Validation")
        validation_expr = st.text_input("Enter validation expression (e.g., `Age > 0 & Age < 120`)")
        if validation_expr:
            try:
                df = df.query(validation_expr)
                st.success("Validation applied successfully!")
            except Exception as e:
                st.error(f"Error in validation: {str(e)}")
        
        st.header("Duplicate Handling")
        duplicates = df.duplicated().sum()
        st.write(f"Number of duplicates: {duplicates}")
        if duplicates > 0 and st.checkbox("Remove duplicates?"):
            df.drop_duplicates(inplace=True)
            st.success("Duplicates removed!")
        
        st.header("Missing Values Management")
        null_counts = df.isnull().sum()
        st.write("Null values per column:")
        st.write(null_counts)
        
        handling_method = st.selectbox("Missing value handling method", 
                                      ['Delete', 'Mean', 'Median', 'Mode', 'KNN', 'Constant'])
        
        if handling_method == 'Delete':
            df.dropna(inplace=True)
        elif handling_method in ['Mean', 'Median', 'Mode']:
            for col in df.select_dtypes(include=np.number).columns:
                if handling_method == 'Mean':
                    df[col].fillna(df[col].mean(), inplace=True)
                elif handling_method == 'Median':
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
        elif handling_method == 'KNN':
            k = st.number_input("Number of neighbors (k)", 3, 10, 5)
            imputer = KNNImputer(n_neighbors=k)
            df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        else:
            constant = st.text_input("Enter constant value")
            df.fillna(constant, inplace=True)
        
        st.header("Outlier Management")
        outlier_method = st.selectbox("Outlier detection method", ['Z-score', 'IQR'])
        
        if df is not None:
            # Work only with numeric columns
            numeric_cols = df.select_dtypes(include=np.number).columns
            outliers = pd.Series(dtype=int)
        
            if outlier_method == 'Z-score':
                if not numeric_cols.empty:
                    z_scores = np.abs(stats.zscore(df[numeric_cols]))
                    outliers = (z_scores > 3).sum(axis=0)
            else:
                if not numeric_cols.empty:
                    Q1 = df[numeric_cols].quantile(0.25)
                    Q3 = df[numeric_cols].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = ((df[numeric_cols] < (Q1 - 1.5 * IQR)) | 
                               (df[numeric_cols] > (Q3 + 1.5 * IQR))).sum(axis=0)
        
            st.write("Outliers per column (numeric columns only):")
            st.write(outliers)
        
            action = st.selectbox("Outlier action", ['Keep', 'Remove', 'Cap'])
            
            if action != 'Keep' and not numeric_cols.empty:
                if action == 'Remove':
                    # Remove outliers using Z-score method
                    if outlier_method == 'Z-score':
                        df = df[(z_scores < 3).all(axis=1)]
                    # Remove outliers using IQR method
                    else:
                        mask = ~((df[numeric_cols] < (Q1 - 1.5 * IQR)) | 
                                (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
                        df = df[mask]
                else:
                    # Cap outliers
                    for col in numeric_cols:
                        lower = df[col].quantile(0.05)
                        upper = df[col].quantile(0.95)
                        df[col] = df[col].clip(lower, upper)
                
                st.session_state.df = df
                st.success("Outliers handled successfully!")
        
        st.header("Data Transformation")
        transform_method = st.selectbox("Select transformation", 
                                      ['Log', 'Standardization', 'Normalization'])
        
        if transform_method == 'Log':
            num_col = st.selectbox("Select numeric column", df.select_dtypes(include=np.number).columns)
            df[num_col] = np.log1p(df[num_col])
        elif transform_method == 'Standardization':
            scaler = StandardScaler()
            df[df.select_dtypes(include=np.number).columns] = scaler.fit_transform(df.select_dtypes(include=np.number))
        else:
            scaler = MinMaxScaler()
            df[df.select_dtypes(include=np.number).columns] = scaler.fit_transform(df.select_dtypes(include=np.number))
        
        st.header("Export Data")
        if st.button("Download Cleaned Data"):
            csv = df.to_csv(index=False)
            st.download_button("Download CSV", csv, "cleaned_data.csv")

else:
    st.info("Please upload data files to begin cleaning")
