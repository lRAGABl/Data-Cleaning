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

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'date_columns' not in st.session_state:
    st.session_state.date_columns = []

def safe_convert_dtypes(df):
    """
    Safely convert DataFrame columns to appropriate types
    to avoid PyArrow serialization issues
    """
    for col in df.columns:
        # Convert object columns to string
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str)
        
        # Handle potential mixed type columns
        try:
            if df[col].dtype == 'float64':
                df[col] = pd.to_numeric(df[col], errors='coerce')
            elif df[col].dtype == 'int64':
                df[col] = pd.to_numeric(df[col], downcast='integer')
        except Exception as e:
            st.warning(f"Could not convert column {col}: {str(e)}")
            df[col] = df[col].astype(str)
    
    return df

def load_data(uploaded_files):
    dfs = []
    for file in uploaded_files:
        try:
            # Use safe type conversion during data loading
            if file.name.endswith('.csv'):
                df = pd.read_csv(file, dtype=str)  # Initially read all as string
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
            # Ensure the dataframe is safely converted before further processing
            df = safe_convert_dtypes(st.session_state.df.copy())
            
            # Rest of your existing code remains the same
            # ... [All previous code remains unchanged]

            # Modify dataframe display to use safe conversion
            st.subheader("First 20 Rows")
            display_df = safe_convert_dtypes(df.head(20))
            st.dataframe(display_df)

            # When exporting, apply safe conversion
            st.header("💾 Export Data")
            if st.button("Download Cleaned Data"):
                if not df.empty:
                    # Apply safe conversion before export
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
