from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import prince
import streamlit as st
from sklearn.decomposition import FactorAnalysis
from factor_analyzer import FactorAnalyzer
from sklearn.preprocessing import StandardScaler
from factor_analyzer.factor_analyzer import (
    calculate_kmo,
    calculate_bartlett_sphericity
)


class FactorStrategy(ABC):
    name: str

    @abstractmethod
    def fit(self, df: pd.DataFrame, n_factors: int):
        pass

class MCAStrategy(FactorStrategy):
    name = "MCA"

    def fit(self, df, n_factors):
        df = df.astype("category")
        model = prince.MCA(n_components=n_factors, random_state=42)
        model.fit(df)
        return model

class ContinuousFAStrategy(FactorStrategy):
    name = "FA"

    def fit(self, df, n_factors):
        df = StandardScaler().fit_transform(df)
        model = FactorAnalysis(n_components=n_factors)
        model.fit(df)
        return model


class PolychoricFAStrategy(FactorStrategy):
    name = "Polychoric FA"

    def fit(self, df, n_factors):

        model = FactorAnalyzer(n_factors=n_factors, method='minres', rotation='promax')
        model.fit(df)

        return model



class FAMDStrategy(FactorStrategy):
    name = "FAMD"

    def fit(self, df, n_factors):
        
        df = df.copy()
        num_cols = []
        cat_cols = []
        print(df.dtypes)
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                num_cols.append(col)
            else:
                try:
                    pd.to_numeric(df[col].dropna().head(100)) # Test sample for speed
                    num_cols.append(col)
                except (ValueError, TypeError):
                    cat_cols.append(col)
            if num_cols:
                df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
        
                df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    
            if cat_cols:
                df[cat_cols] = df[cat_cols].astype('object')
                df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])


        model = prince.FAMD(
            n_components=n_factors,
            n_iter=3,
            copy=True,
            check_input=True,
            random_state=42,
            engine="sklearn",
            handle_unknown="error" 
        )
        
        model.fit(df)

        return model




def check_factor_adequacy(df: pd.DataFrame, alpha=0.05):
    chi2, p_value = calculate_bartlett_sphericity(df)
    _, kmo_model = calculate_kmo(df)

    return {
        "kmo": kmo_model,
        "bartlett_chi2": chi2,
        "bartlett_p_value": p_value,
        "is_factorable": (kmo_model >= 0.6) and (p_value < alpha)
    }

def select_rotation(corr_matrix, threshold=0.3):
    off_diag = corr_matrix - np.eye(corr_matrix.shape[0])
    max_corr = np.abs(off_diag).max()
    return "oblimin" if max_corr > threshold else "varimax"


# Strategy Selector
def select_strategy(df: pd.DataFrame) -> FactorStrategy:
    
    all_cols = set(df.columns)
    num_cols = set(df.select_dtypes(include=[np.number]).columns)
    cat_cols = set(df.select_dtypes(include=['object', 'category', 'bool']).columns)
    
    ordinal_cols = {c for c in num_cols if 2 <= df[c].nunique() <= 10}
    continuous_cols = num_cols - ordinal_cols

    # Case: Purely Continuous
    if len(num_cols) == len(all_cols) and len(ordinal_cols) == 0:
        return ContinuousFAStrategy() # Uses Pearson Correlation
    
    # Case: Purely Ordinal (or Binary)
    elif len(ordinal_cols) == len(all_cols):
        return PolychoricFAStrategy() # Uses Polychoric/Tetrachoric Correlation
    
    # Case: Purely Nominal (Categorical)
    elif len(cat_cols) == len(all_cols):
        return MCAStrategy() # Multiple Correspondence Analysis

    # Case: NO Continuous variables
    # This prevents the "All variables are qualitative" error in FAMD
    elif len(continuous_cols) == 0:
        return MCAStrategy() 

    # Case: Truly Mixed Data (Categorical/Ordinal AND Continuous)
    else:
        return FAMDStrategy()


def export_loadings(model):
    """
    Returns a tidy DataFrame of factor loadings, handling different library backends.
    """
    # 1. Identify the source of the loadings based on the model type
    if hasattr(model, 'loadings_'): 
        # factor_analyzer
        loadings = model.loadings_
    elif hasattr(model, 'components_'): 
        # sklearn
        loadings = model.components_.T
    elif hasattr(model, 'column_correlations_'): 
        # prince (MCA/FAMD)
        loadings = model.column_correlations_.values
    else:
        raise ValueError("Model type not recognized for loading extraction.")

    factors = [f"{factor_prefix}{i+1}" for i in range(loadings.shape[1])]

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

def get_mca_metrics(df):
   
    mca = prince.MCA(n_components=min(len(df.columns), 10))
    mca = mca.fit(df)
    K = len(df.columns)
    raw_ev = mca.eigenvalues_
    
    threshold = 1/K
    corrected_ev = [((K/(K-1)) * (ev - threshold))**2 if ev > threshold else 0 for ev in raw_ev]
    
    suggested_mca = 1
    for i in range(1, len(corrected_ev)):
        if corrected_ev[i-1] - corrected_ev[i] < (sum(corrected_ev) * 0.05):
            suggested_mca = i
            break
            
    cumulative = np.cumsum(corrected_ev) / sum(corrected_ev)
    dims_for_90 = np.argmax(cumulative >= 0.90) + 1
    
    return suggested_mca, dims_for_90

def get_famd_metrics(df):
    df_temp = df.copy()
    print(df_temp.dtypes)

    num_cols = df_temp.select_dtypes(include=[np.number]).columns
    ordinal_cols = [c for c in num_cols if 2 <= df_temp[c].nunique() <= 10]
    continuous_cols = num_cols.difference(ordinal_cols)

    for col in ordinal_cols:
        df_temp[col] = df_temp[col].astype("category")
    for col in continuous_cols:
        df_temp[col] = df_temp[col].astype(float)
    cat_cols = df_temp.select_dtypes(include=["object", "category", "bool"]).columns
    for col in cat_cols:
        df_temp[col] = df_temp[col].astype(str)
    if len(cat_cols) == 0:
        corr_matrix = df[num_cols].corr()
        eigenvalues = np.linalg.eigvals(corr_matrix)
        return int(np.sum(eigenvalues > 1))

    
    n_comp = min(len(df_temp.columns), 10)
    famd = prince.FAMD(n_components=n_comp, random_state=42)
    famd.fit(df_temp)

    var_col = famd.eigenvalues_summary["% of variance"]
    if var_col.dtype == object:
        inertia = (
            var_col.str.replace("%", "", regex=False)
            .astype(float)
            .values / 100
        )
    else:
        inertia = var_col.values / 100
    
    cumulative_inertia = np.cumsum(inertia)
    idx = np.where(cumulative_inertia >= 0.70)[0] # 70% inertia rule
    return int(idx[0] + 1) if len(idx) > 0 else n_comp


def get_kaiser_famd(df):
    df_temp = df.copy()

    num_cols = df_temp.select_dtypes(include=[np.number]).columns
    ordinal_cols = [c for c in num_cols if 2 <= df_temp[c].nunique() <= 10]

    for col in ordinal_cols:
        df_temp[col] = df_temp[col].astype("category")
    for col in num_cols.difference(ordinal_cols):
        df_temp[col] = df_temp[col].astype(float)
    cat_cols = df_temp.select_dtypes(include=["object", "category", "bool"]).columns
    for col in cat_cols:
        df_temp[col] = df_temp[col].astype(str)
    if len(cat_cols) == 0:
        corr_matrix = df[num_cols].corr()
        eigenvalues = np.linalg.eigvals(corr_matrix)
        return int(np.sum(eigenvalues > 1))

    famd = prince.FAMD(n_components=min(df_temp.shape), random_state=42)
    famd.fit(df_temp)

    return int(np.sum(famd.eigenvalues_ > 1.0))


# Strategy description
def strategy(strategy_name):
    if strategy_name == 'MCA':
        return "Your dataset has nominal variable or no continuous variable, we will proceed with the multiple correspondence analysis method"
    elif strategy_name == 'FA':
        return "All your variable are continuous, we will proceed with the factor analysis"
    elif strategy_name == "Polychoric FA":
        return "Your dataset has ordinal variable, we will proceed with the polychoric factor analysis"
    else : 
        return "Your dataset has mixed variable, we will proceed with the FAMD method"

        


