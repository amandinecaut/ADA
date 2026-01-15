from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import prince
import pingouin as pg

from sklearn.decomposition import FactorAnalysis
from factor_analyzer import FactorAnalyzer
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
        model = prince.MCA(n_components=n_factors, random_state=42)
        model.fit(df)
        return model

class ContinuousFAStrategy(FactorStrategy):
    name = "FA"

    def fit(self, df, n_factors):
        model = FactorAnalysis(n_components=n_factors)
        model.fit(df)
        return model


class PolychoricFAStrategy(FactorStrategy):
    name = "Polychoric FA"

    def fit(self, df, n_factors):
        adequacy = check_factor_adequacy(df)
        if not adequacy["is_factorable"]:
            raise ValueError(f"Data not factorable: {adequacy}")

        # Polychoric correlation (Pingouin)
        corr = pg.pcorr(df).values

        # Preliminary oblique FA (to test factor correlations)
        temp_fa = FactorAnalyzer(
            n_factors=n_factors,
            rotation="oblimin",
            is_corr_matrix=True
        )
        temp_fa.fit(corr)

        factor_corr = temp_fa.get_factor_correlation_matrix()

        rotation = select_rotation(factor_corr)

        # Final FA
        model = FactorAnalyzer(
            n_factors=n_factors,
            rotation=rotation,
            is_corr_matrix=True
        )
        model.fit(corr)

        model.adequacy_ = adequacy
        model.rotation_ = rotation

        return model


class FAMDStrategy(FactorStrategy):
    name = "FAMD"

    def fit(self, df, n_factors):
        df = df.copy()
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].astype(float)

        cat_cols = df.select_dtypes(exclude=[np.number]).columns
        for col in cat_cols:
            df[col] = df[col].astype(str)

        model = prince.FAMD(n_components=n_factors, random_state=42)
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
    
    # Identify ordinal candidates (numeric but few unique values)
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
    
    # Case: Mixed Data 
    else:
        return FAMDStrategy() 


def export_loadings(model, feature_names, factor_prefix="Factor"):
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
    
    num_cols = df_temp.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df_temp[col] = df_temp[col].astype(float)
        
    cat_cols = df_temp.select_dtypes(exclude=[np.number]).columns
    for col in cat_cols:
        df_temp[col] = df_temp[col].astype(str)


    n_comp = min(len(df_temp.columns), 10)
    famd = prince.FAMD(n_components=n_comp, random_state=42)
    famd = famd.fit(df_temp)
    

    var_col = famd.eigenvalues_summary['% of variance']
    
    clean_variance = (
        var_col.astype(str)
        .str.replace('%', '', regex=False)
        .astype(float)
    )
    

    cumulative_inertia = np.cumsum(clean_variance / 100)
    passing_indices = np.where(cumulative_inertia >= 0.70)[0]
    
    if len(passing_indices) > 0:
        famd_suggested = int(passing_indices[0] + 1)
    else:
        famd_suggested = n_comp

    return famd_suggested

def get_kaiser_famd(df):
    df_temp = df.copy()
    num_cols = df_temp.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df_temp[col] = df_temp[col].astype(float)
        
    cat_cols = df_temp.select_dtypes(exclude=[np.number]).columns
    for col in cat_cols:
        df_temp[col] = df_temp[col].astype(str)
    famd = prince.FAMD(n_components=min(df_temp.shape))
    famd.fit(df_temp)
    ev = famd.eigenvalues_
       
    return len([i for i in ev if i > 1.0])

# Strategy description
def strategy(strategy_name):
    if strategy_name == 'MCA':
        return "Your dataset has nominal variable, we will proceed with the multiple correspondence analysis method"
    elif strategy_name == 'FA':
        return "All your variable are continuous, we will proceed with the factor analysis"
    elif strategy_name == "Polychoric FA":
        return "Your dataset has ordinal variable, we will proceed with the polychoric factor analysis"
    else : 
        return "Your dataset has mixed variable, we will proceed with the FAMD method"

        


