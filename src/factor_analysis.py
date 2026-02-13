from abc import ABC, abstractmethod
from pyexpat import model
from narwhals import col
import numpy as np
import pandas as pd
import prince
from pyparsing import col
import streamlit as st
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler

class FactorStrategy(ABC):
    name: str

    @abstractmethod
    def fit(self, df: pd.DataFrame, n_factors: int):
        pass

class ContinuousFAStrategy(FactorStrategy):
    """Path 1: Standard Factor Analysis for continuous numerical data."""
    name = "FA"

    def fit(self, df, n_factors):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)
        
        model = FactorAnalysis(n_components=n_factors)
        scores = model.fit_transform(scaled_data)
        
        return model, scores
    

class FAMDStrategy(FactorStrategy):
    """Path 2: Factor Analysis of Mixed Data for datasets with text/categories and numbers."""
    name = "FAMD"

    def fit(self, df, n_factors):
        df_famd = df.copy()
        
        num_cols = df_famd.select_dtypes(include=[np.number]).columns
    
        if len(num_cols) == 0:
            df_famd['__dummy_numeric__'] = 1.0
        
      
        original_columns = df_famd.columns.tolist()
        
        for col in original_columns:
           
            if pd.api.types.is_numeric_dtype(df_famd[col]):
                df_famd[col] = df_famd[col].astype(float).fillna(0)
               
            else:
                
                dummies = pd.get_dummies(df_famd[col], prefix=col, dtype=int)
                dummies.columns = ['the ' + col.replace('_', ' ').lower() + ' is ' + dummy_col.replace(col + '_', '').replace('_', ' ').lower() for dummy_col in dummies.columns]
                df_famd = pd.concat([df_famd, dummies], axis=1)
                df_famd.drop(columns=[col], inplace=True)
        df_famd.columns = df_famd.columns.str.replace('_', ' ').str.lower()
       



        st.session_state.features = [col for col in original_columns if col != '__dummy_numeric__']
        st.session_state.features_famd = df_famd.columns.tolist()  
        st.session_state.df_famd = df_famd
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_famd)

        
        model = FactorAnalysis(n_components=n_factors)
        scores = model.fit_transform(scaled_data)
        
        return model, scores
    



class FAMDStrategy_v1(FactorStrategy):
    """Path 2: Factor Analysis of Mixed Data for datasets with text/categories and numbers."""
    name = "FAMD"

    def fit(self, df, n_factors):
        df_famd = df.copy()
        
        num_cols = df_famd.select_dtypes(include=[np.number]).columns
    
        if len(num_cols) == 0:
            df_famd['__dummy_numeric__'] = 1.0
        
        # Store original column names BEFORE processing
        original_columns = df_famd.columns.tolist()
        
        for col in df_famd.columns:
            if pd.api.types.is_numeric_dtype(df_famd[col]):
                df_famd[col] = df_famd[col].astype(float).fillna(0)

            else:
                df_famd[col] = df_famd[col].astype(object).fillna("Unknown")
                dummies = pd.get_dummies(df_famd[col], prefix=col, dtype=int)
                dummies.columns = [c.replace('_', ' ').lower() for c in dummies.columns]
                df_famd = pd.concat([df_famd, dummies], axis=1)
                df_famd.drop(columns=[col], inplace=True)
        

        st.session_state.features = [col for col in original_columns if col != '__dummy_numeric__']
        st.session_state.features_famd = df_famd.columns.tolist()  
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_famd)

        model = FactorAnalysis(n_components=n_factors)
        scores = model.fit_transform(scaled_data)
        
        return model, scores
    



def select_strategy(df: pd.DataFrame) -> FactorStrategy:
    """
    Automatically chooses FA if data is purely numerical, 
    otherwise chooses FAMD for mixed types.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns
    
    # If all columns are numerical, use standard Factor Analysis
    if len(num_cols) == len(df.columns):
        return ContinuousFAStrategy()
    
    # Otherwise, use FAMD for mixed (text and numbers)
    return FAMDStrategy()

def export_loadings(model, feature_names):
    """
    Returns a tidy DataFrame of factor loadings.
    """
    if hasattr(model, 'components_'): 
        # sklearn FactorAnalysis
        loadings = model.components_.T
    elif hasattr(model, 'column_correlations_'): 
        # prince FAMD
        loadings = model.column_correlations_.values
    else:
        raise ValueError("Model type not recognized.")

    factors = [f"Factor_{i+1}" for i in range(loadings.shape[1])]

    df_loadings = (
        pd.DataFrame(loadings, index=feature_names, columns=factors)
        .reset_index()
        .rename(columns={"index": "variable"})
        .melt(id_vars="variable", var_name="factor", value_name="loading")
        .sort_values(by=["factor", "loading"], ascending=[True, False])
    )
    return df_loadings

# Opitimal number of factors
def get_kaiser_criterion():
    x = st.session_state.df_filtered.loc[:, st.session_state.features].values
    x = StandardScaler().fit_transform(x)
    corr_matrix =  np.corrcoef(x, rowvar=False)
    eigenvalues = np.linalg.eigh(corr_matrix)[0] 
    kaiser_n = eigenvalues[eigenvalues > 1]
    return len(kaiser_n)

def HornParallelAnalysis(K=10):

    data = st.session_state.df_filtered.loc[:, st.session_state.features].values
    data = StandardScaler().fit_transform(data)
    n, m = data.shape

    def get_ev(d):
        corr_mtx = np.corrcoef(d, rowvar=False)
        ev = np.linalg.eigvalsh(corr_mtx)
        return np.sort(ev)[::-1] # Sort descending
    dataEv = get_ev(data)

    sumRandomEigens = np.zeros(m)
    for runNum in range(K):
        randomData = np.random.normal(size=(n, m))
        sumRandomEigens += get_ev(randomData)

    avgRandomEigens = sumRandomEigens / K
    suggestedFactors = np.sum(dataEv > avgRandomEigens)

    return suggestedFactors

def get_famd_metrics(df: pd.DataFrame):
    """
    Heuristic to suggest number of factors for FAMD.
    Fixes the 'All variables are qualitative' error.
    """
    df_temp = df.copy()
   
    # Identify types
    num_cols = df_temp.select_dtypes(include=[np.number]).columns
    
    
    # FIX: Add dummy numeric column if none exist to prevent the ValueError
    if len(num_cols) == 0:
        df_temp['__dummy_numeric__'] = 1.0
        
   
    for col in df_temp.columns:
        if pd.api.types.is_numeric_dtype(df_temp[col]):
            df_temp[col] = df_temp[col].astype(float).fillna(0)
        else:
            df_temp[col] = df_temp[col].astype(object).fillna("Unknown")

    n_comp = min(len(df_temp.columns), 10)
    famd = prince.FAMD(n_components=n_comp, random_state=42, check_input=False)
    famd.fit(df_temp)
        
    var_col = famd.eigenvalues_summary["% of variance"]
    
    if var_col.dtype == object:
        inertia = var_col.str.replace("%", "", regex=False).astype(float).values / 100
    else:
        inertia = var_col.values / 100
        
    cumulative_inertia = np.cumsum(inertia)
    idx = np.where(cumulative_inertia >= 0.70)[0]
    return int(idx[0] + 1) if len(idx) > 0 else n_comp


def get_kaiser_famd(df):
    df_temp = df.copy()
    num_cols = df_temp.select_dtypes(include=[np.number]).columns
    
    if len(num_cols) == 0:
        df_temp['__dummy_numeric__'] = 1.0
        
    # Standardize data types for Prince
    for col in df_temp.columns:
        if pd.api.types.is_numeric_dtype(df_temp[col]):
            df_temp[col] = df_temp[col].astype(float).fillna(0)
        else:
            df_temp[col] = df_temp[col].astype(object).fillna("Unknown")

    famd = prince.FAMD(n_components=min(df_temp.shape), random_state=42)
    famd.fit(df_temp)

    return int(np.sum(famd.eigenvalues_ > 1.0))

def get_strategy_description(strategy_name):
    if strategy_name == 'FA':
        return "All your variables are numerical. We will proceed with standard Factor Analysis."
    else: 
        return "Your dataset contains mixed variables (text and numbers). We will proceed with FAMD."