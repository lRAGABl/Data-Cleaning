import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats
import io
from datetime import datetime
import pyarrow as pa

st.set_page_config(layout="wide")

if 'df' not in st.session_state:
    st.session_state.df = None
if 'date_columns' not in st.session_state:
    st.session_state.date_columns = []

def safe_convert_dtypes(df):
    """
    Safely convert DataFrame columns to appropriate types
    to avoid PyArrow serialization issues and deprecation warnings
    """
    # Create a deep copy to avoid SettingWithCopyWarning
    df_copy = df.copy(deep=True)
    
    for col in df_copy.columns:
        try:
            # Handle object (string) columns
            if df_copy[col].dtype == 'object':
                try:
                    # Try to convert to numeric, catching any conversion errors
                    numeric_converted = pd.to_numeric(df_copy[col])
                    
                    # If conversion was successful and not all NaN
                    if not numeric_converted.isna().all():
                        df_copy.loc[:, col] = numeric_converted
                    else:
                        # If conversion results in all NaN, convert to string
                        df_copy.loc[:, col] = df_copy[col].astype(str)
                except (ValueError, TypeError):
                    # If numeric conversion fails, convert to string
                    df_copy.loc[:, col] = df_copy[col].astype(str)
            
            # Handle float columns
            elif df_copy[col].dtype == 'float64':
                try:
                    df_copy.loc[:, col] = pd.to_numeric(df_copy[col], downcast='float')
                except (ValueError, TypeError):
                    df_copy.loc[:, col] = df_copy[col].astype(str)
            
            # Handle integer columns
            elif df_copy[col].dtype == 'int64':
                try:
                    df_copy.loc[:, col] = pd.to_numeric(df_copy[col], downcast='integer')
                except (ValueError, TypeError):
                    df_copy.loc[:, col] = df_copy[col].astype(str)
        
        except Exception as e:
            st.warning(f"Could not convert column {col}: {str(e)}")
            # Fallback to string conversion
            df_copy.loc[:, col] = df_copy[col].astype(str)
    
    return df_copy

def load_data(uploaded_files):
    dfs = []
    for file in uploaded_files:
        try:
            # Use safe type conversion during data loading
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)  # Remove dtype=str
            elif file.name.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file)
            elif file.name.endswith('.json'):
                df = pd.read_json(file)
            
            # Apply safe type conversion
            df = safe_convert_dtypes(df)
            
            dfs.append(df)
        except Exception as e:
            st.error(f"Error loading {file.name}: {str(e)}")
    
    # Combine or return single dataframe
    return pd.concat(dfs) if len(dfs) > 1 else dfs[0] if dfs else None

def load_data(uploaded_files):
    dfs = []
    for file in uploaded_files:
        try:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file, dtype=str)
            elif file.name.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file)
            elif file.name.endswith('.json'):
                df = pd.read_json(file)
            
            df = safe_convert_dtypes(df)
            
            dfs.append(df)
        except Exception as e:
            st.error(f"Error loading {file.name}: {str(e)}")
    
    return pd.concat(dfs) if len(dfs) > 1 else dfs[0] if dfs else None

def validate_date(date_str, format):
    try:
        datetime.strptime(str(date_str), format)
        return True
    except ValueError:
        return False

def main():
    st.sidebar.header("Data Upload")
    uploaded_files = st.sidebar.file_uploader("Upload datasets", 
                                            accept_multiple_files=True, 
                                            type=['csv', 'xlsx', 'json'])
    
    if uploaded_files:
        if st.session_state.df is None:
            st.session_state.df = load_data(uploaded_files)
        
        if st.session_state.df is not None:
            df = safe_convert_dtypes(st.session_state.df.copy())
            
            st.header("📅 Date Validation")
            date_cols = st.multiselect("Select potential date columns", df.columns)
            
            if date_cols:
                date_format = st.text_input("Enter expected date format (e.g., %Y-%m-%d)")
                if date_format:
                    invalid_dates = {}
                    for col in date_cols:
                        invalid = df[col].apply(lambda x: not validate_date(str(x), date_format) if pd.notnull(x) else True)
                        invalid_dates[col] = df[col][invalid].index.tolist()
                    
                    if any(invalid_dates.values()):
                        st.error("Invalid dates found:")
                        for col, indices in invalid_dates.items():
                            if indices:
                                st.write(f"Column '{col}': {len(indices)} invalid dates at rows {indices[:10]}{'...' if len(indices)>10 else ''}")
                        if st.button("Convert to proper datetime format"):
                            try:
                                for col in date_cols:
                                    df[col] = pd.to_datetime(df[col], errors='coerce', format=date_format)
                                st.session_state.df = df
                                st.success("Date conversion completed!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Date conversion failed: {str(e)}")
                    else:
                        st.success("All dates are valid!")
            
            st.header("🔍 Data Overview")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Dataset Info")
                buffer = io.StringIO()
                df.info(buf=buffer)
                st.text(buffer.getvalue())
                
            with col2:
                st.subheader("Basic Statistics")
                st.write(df.describe(include='all'))
            
            st.subheader("First 20 Rows")
            display_df = safe_convert_dtypes(df.head(20))
            st.dataframe(display_df)

            st.header("📝 Column Management")
            selected_col = st.selectbox("Select column to rename", df.columns)
            new_name = st.text_input("New column name", selected_col)
            if st.button("Rename Column"):
                df.rename(columns={selected_col: new_name}, inplace=True)
                st.session_state.df = df
                st.rerun()

            st.subheader("🔒 Data Validation")
            validation_expr = st.text_input("Enter validation expression (e.g., `Age > 0 & Age < 120`)")
            if validation_expr:
                try:
                    df = df.query(validation_expr)
                    st.session_state.df = df
                    st.success("Validation applied successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error in validation: {str(e)}")
            
            st.header("♻️ Duplicate Handling")
            duplicates = df.duplicated().sum()
            st.write(f"Number of duplicates: {duplicates}")
            if duplicates > 0 and st.checkbox("Remove duplicates?"):
                df.drop_duplicates(inplace=True)
                st.session_state.df = df
                st.success("Duplicates removed!")
                st.rerun()
            
            st.header("❓ Missing Values Management")
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
            
            st.session_state.df = df
            
            st.header("📊 Outlier Management")
            outlier_method = st.selectbox("Outlier detection method", ['Z-score', 'IQR'])
            
            numeric_cols = df.select_dtypes(include=np.number).columns
            outliers = pd.Series(dtype=int)
        
            if outlier_method == 'Z-score' and not numeric_cols.empty:
                z_scores = np.abs(stats.zscore(df[numeric_cols]))
                outliers = (z_scores > 3).sum(axis=0)
            elif not numeric_cols.empty:
                Q1 = df[numeric_cols].quantile(0.25)
                Q3 = df[numeric_cols].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df[numeric_cols] < (Q1 - 1.5 * IQR)) | 
                          (df[numeric_cols] > (Q3 + 1.5 * IQR))).sum(axis=0)
            
            if not numeric_cols.empty:
                st.write("Outliers per column (numeric columns only):")
                st.write(outliers)
            
                action = st.selectbox("Outlier action", ['Keep', 'Remove', 'Cap'])
                
                if action != 'Keep':
                    if action == 'Remove':
                        if outlier_method == 'Z-score':
                            df = df[(z_scores < 3).all(axis=1)]
                        else:
                            mask = ~((df[numeric_cols] < (Q1 - 1.5 * IQR)) | 
                                    (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
                            df = df[mask]
                    else:
                        for col in numeric_cols:
                            lower = df[col].quantile(0.05)
                            upper = df[col].quantile(0.95)
                            df[col] = df[col].clip(lower, upper)
                    
                    st.session_state.df = df
                    st.success("Outliers handled successfully!")
                    st.rerun()
            
            st.header("🔄 Data Transformation")
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            
            if numeric_cols:
                transform_method = st.selectbox(
                    "Select transformation type",
                    ['Log Transform', 'Standardization', 'Normalization']
                )

                if transform_method == 'Log Transform':
                    selected_cols = [st.selectbox("Select column for log transform", numeric_cols)]
                else:
                    selected_cols = st.multiselect("Select columns to transform", numeric_cols, default=numeric_cols)
        
                if selected_cols:
                    try:
                        if transform_method == 'Log Transform':
                            df[selected_cols] = np.log1p(df[selected_cols])
                        elif transform_method == 'Standardization':
                            scaler = StandardScaler()
                            df[selected_cols] = scaler.fit_transform(df[selected_cols])
                        else:
                            scaler = MinMaxScaler()
                            df[selected_cols] = scaler.fit_transform(df[selected_cols])
                        
                        st.session_state.df = df
                        st.success(f"{transform_method} applied successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Transformation failed: {str(e)}")
            
            st.header("💾 Export Data")
            if st.button("Download Cleaned Data"):
                if not df.empty:
                    export_df = safe_convert_dtypes(df)
                    csv = export_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="cleaned_data.csv",
                        mime="text/csv",
                    )
                else:
                    st.error("Cannot download empty dataset!")

    else:
        st.info("Please upload data files to begin cleaning")

if __name__ == "__main__":
    main()
