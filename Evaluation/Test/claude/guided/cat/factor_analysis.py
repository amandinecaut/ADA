"""
Factor Analysis on Cat Breeds Dataset
This script performs factor analysis to identify underlying latent factors
in cat breed characteristics.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================

# Load the CSV data
df = pd.read_csv('cat_breeds.csv')

# Load the column mappings
column_map = pd.read_excel('map.xlsx')
column_descriptions = dict(zip(column_map['Key'], column_map['Value']))

print("=" * 80)
print("FACTOR ANALYSIS ON CAT BREEDS DATASET")
print("=" * 80)
print(f"\nDataset shape: {df.shape}")
print(f"Number of breeds: {len(df)}")

# ============================================================================
# 2. SELECT NUMERIC FEATURES FOR ANALYSIS
# ============================================================================

# Select only numeric columns for factor analysis (exclude name, length, origin)
numeric_cols = [
    'min_life_expectancy', 'max_life_expectancy', 'min_weight', 'max_weight',
    'family_friendly', 'shedding', 'general_health', 'playfulness',
    'children_friendly', 'grooming', 'intelligence', 'other_pets_friendly'
]

# Create a subset with numeric features
df_numeric = df[numeric_cols].copy()

print(f"\nNumeric features selected: {len(numeric_cols)}")
print("Features included:")
for col in numeric_cols:
    if col in column_descriptions:
        print(f"  - {col}: {column_descriptions[col]}")

# Handle missing values
print(f"\nMissing values: {df_numeric.isnull().sum().sum()}")
df_numeric = df_numeric.dropna()

print(f"Dataset shape after removing NaNs: {df_numeric.shape}")

# ============================================================================
# 3. STANDARDIZE THE DATA
# ============================================================================

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_numeric)
df_scaled = pd.DataFrame(df_scaled, columns=numeric_cols)

print("\nData standardized (mean=0, std=1)")

# ============================================================================
# 4. TEST FACTORABILITY OF THE DATA
# ============================================================================

print("\n" + "=" * 80)
print("TESTING DATA FACTORABILITY")
print("=" * 80)

# Bartlett's Test of Sphericity (manual implementation)
def calculate_bartlett_sphericity(data):
    """Calculate Bartlett's test of sphericity"""
    n, p = data.shape
    corr_matrix = np.corrcoef(data.T)
    det = np.linalg.det(corr_matrix)
    chi_square = -((n - 1) - (2*p + 5)/6) * np.log(det)
    df = p * (p - 1) / 2
    p_value = 1 - chi2.cdf(chi_square, df)
    return chi_square, p_value

# Calculate Bartlett's test
chi_square_value, p_value = calculate_bartlett_sphericity(df_scaled.values)
print(f"\nBartlett's Test of Sphericity:")
print(f"  Chi-square value: {chi_square_value:.4f}")
print(f"  P-value: {p_value:.4e}")
print(f"  Interpretation: {'Data is suitable for FA' if p_value < 0.05 else 'Data may not be suitable for FA'}")

# Calculate KMO (Kaiser-Meyer-Olkin) - manual implementation
def calculate_kmo(data):
    """Calculate Kaiser-Meyer-Olkin (KMO) measure"""
    corr_matrix = np.corrcoef(data.T)
    p = corr_matrix.shape[0]
    
    # Partial correlations
    try:
        inv_corr = np.linalg.inv(corr_matrix)
        partial_corr = -inv_corr / np.sqrt(np.outer(np.diag(inv_corr), np.diag(inv_corr)))
        np.fill_diagonal(partial_corr, 0)
    except:
        partial_corr = np.zeros_like(corr_matrix)
    
    rij_sq = corr_matrix ** 2
    pij_sq = partial_corr ** 2
    
    kmo_per_var = np.sum(rij_sq, axis=0) / (np.sum(rij_sq, axis=0) + np.sum(pij_sq, axis=0) + 1e-10)
    kmo_model = np.mean(kmo_per_var)
    
    return kmo_per_var, kmo_model

kmo_all, kmo_model = calculate_kmo(df_scaled.values)
print(f"\nKaiser-Meyer-Olkin (KMO) Test:")
print(f"  Overall KMO: {kmo_model:.4f}")
print(f"  Interpretation: ", end="")
if kmo_model >= 0.9:
    print("Marvelous")
elif kmo_model >= 0.8:
    print("Meritorious")
elif kmo_model >= 0.7:
    print("Middling")
elif kmo_model >= 0.6:
    print("Mediocre")
elif kmo_model >= 0.5:
    print("Miserable")
else:
    print("Unacceptable")

# ============================================================================
# 5. DETERMINE OPTIMAL NUMBER OF FACTORS
# ============================================================================

print("\n" + "=" * 80)
print("DETERMINING OPTIMAL NUMBER OF FACTORS")
print("=" * 80)

# Use eigenvalues from correlation matrix to determine the number of factors
corr_matrix = np.corrcoef(df_scaled.values.T)
eigenvalues = np.linalg.eigvals(corr_matrix)
eigenvalues = np.sort(eigenvalues)[::-1]  # Sort in descending order

print(f"\nEigenvalues (variance explained by each factor):")
for i, ev in enumerate(eigenvalues[:min(8, len(eigenvalues))]):
    cum_variance = np.sum(eigenvalues[:i+1]) / np.sum(eigenvalues)
    print(f"  Factor {i+1}: {ev:.4f} (Cumulative variance: {cum_variance*100:.2f}%)")

# Kaiser criterion: Keep factors with eigenvalues > 1
n_factors_kaiser = np.sum(eigenvalues > 1)
print(f"\nUsing Kaiser criterion (eigenvalue > 1):")
print(f"  Optimal number of factors: {n_factors_kaiser}")

# 80% cumulative variance explained
cumsum_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)
n_factors_80 = np.argmax(cumsum_variance >= 0.80) + 1
print(f"\nFor 80% cumulative variance explained:")
print(f"  Optimal number of factors: {n_factors_80}")

# Use Kaiser criterion
optimal_n_factors = n_factors_kaiser
print(f"\n{'*' * 80}")
print(f"SELECTED OPTIMAL NUMBER OF FACTORS: {optimal_n_factors}")
print(f"{'*' * 80}")

# ============================================================================
# 6. PERFORM FACTOR ANALYSIS WITH OPTIMAL NUMBER OF FACTORS
# ============================================================================

print("\n" + "=" * 80)
print(f"PERFORMING FACTOR ANALYSIS WITH {optimal_n_factors} FACTORS")
print("=" * 80)

# Perform FA with optimal number of factors using sklearn
fa = FactorAnalysis(n_components=optimal_n_factors, random_state=42, max_iter=1000)
fa.fit(df_scaled.values)

# Get factor loadings
loadings = fa.components_.T  # Transpose to get features x factors
loadings_df = pd.DataFrame(
    loadings,
    columns=[f'Factor {i+1}' for i in range(optimal_n_factors)],
    index=numeric_cols
)

print(f"\nFactor Loadings (after Varimax rotation):")
print(loadings_df.round(3))

# ============================================================================
# 7. NAME THE FACTORS BASED ON LOADINGS
# ============================================================================

print("\n" + "=" * 80)
print("FACTOR INTERPRETATION AND NAMING")
print("=" * 80)

factor_names = {}
for factor_idx in range(optimal_n_factors):
    print(f"\n{'*' * 80}")
    print(f"Factor {factor_idx + 1}:")
    print(f"{'*' * 80}")
    
    # Get the top loading features (by absolute value)
    factor_loadings = loadings_df[f'Factor {factor_idx + 1}'].abs().sort_values(ascending=False)
    
    print(f"\nTop contributing features:")
    for i, (feature, loading_abs) in enumerate(factor_loadings.head(5).items(), 1):
        actual_loading = loadings_df.loc[feature, f'Factor {factor_idx + 1}']
        print(f"  {i}. {feature}: {actual_loading:.3f}")
        if feature in column_descriptions:
            print(f"     ({column_descriptions[feature]})")
    
    # Calculate variance explained by this factor
    variance_explained = np.sum(loadings[:, factor_idx] ** 2) / len(numeric_cols)
    print(f"\nVariance explained by this factor: {variance_explained*100:.2f}%")
    
    # Name the factor based on dominant features
    top_features = factor_loadings.head(3).index.tolist()
    
    if factor_idx == 0:
        if 'family_friendly' in top_features or 'children_friendly' in top_features or 'playfulness' in top_features:
            if 'shedding' in top_features or 'grooming' in top_features:
                factor_names[factor_idx + 1] = "Social & Maintenance Needs"
            else:
                factor_names[factor_idx + 1] = "Social & Behavioral Traits"
        else:
            factor_names[factor_idx + 1] = f"Factor {factor_idx + 1}: {', '.join(top_features[:2])}"
    
    elif factor_idx == 1:
        if 'min_weight' in top_features or 'max_weight' in top_features:
            if 'min_life_expectancy' in top_features or 'max_life_expectancy' in top_features:
                factor_names[factor_idx + 1] = "Size & Lifespan"
            else:
                factor_names[factor_idx + 1] = "Physical Size"
        else:
            factor_names[factor_idx + 1] = f"Factor {factor_idx + 1}: {', '.join(top_features[:2])}"
    
    elif factor_idx == 2:
        if 'intelligence' in top_features or 'playfulness' in top_features:
            factor_names[factor_idx + 1] = "Intelligence & Engagement"
        elif 'general_health' in top_features:
            factor_names[factor_idx + 1] = "Health & Wellness"
        else:
            factor_names[factor_idx + 1] = f"Factor {factor_idx + 1}: {', '.join(top_features[:2])}"
    
    else:
        factor_names[factor_idx + 1] = f"Factor {factor_idx + 1}: {', '.join(top_features[:2])}"
    
    print(f"\n➜ Factor Name: {factor_names[factor_idx + 1]}")

# ============================================================================
# 8. CALCULATE FACTOR SCORES
# ============================================================================

print("\n" + "=" * 80)
print("FACTOR SCORES FOR EACH BREED")
print("=" * 80)

# Calculate factor scores
factor_scores = fa.transform(df_scaled.values)
factor_scores_df = pd.DataFrame(
    factor_scores,
    columns=[f'Factor {i+1}' for i in range(optimal_n_factors)],
    index=df['name'][:len(factor_scores)]
)

# Rename columns with factor names
rename_dict = {f'Factor {k}': v for k, v in factor_names.items()}
factor_scores_df = factor_scores_df.rename(columns=rename_dict)

print(f"\nFactor scores (first 10 breeds):")
print(factor_scores_df.head(10).round(3))

# ============================================================================
# 9. SUMMARY STATISTICS
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

print(f"\nTotal Variance Explained:")
# Calculate explained variance for sklearn's FA
explained_var = np.var(fa.transform(df_scaled.values), axis=0)
total_variance = np.sum(explained_var) / np.sum(np.var(df_scaled.values, axis=0))
print(f"  {total_variance*100:.2f}% of total variance explained by {optimal_n_factors} factors")

print(f"\nVariance Explained by Each Factor:")
factor_variances_pct = explained_var / np.sum(explained_var)
for i, var in enumerate(factor_variances_pct):
    factor_name = factor_names.get(i+1, f"Factor {i+1}")
    print(f"  {factor_name}: {var*100:.2f}%")

# ============================================================================
# 10. VISUALIZATION
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Scree Plot (Eigenvalues)
ax1 = axes[0, 0]
ax1.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'bo-', linewidth=2, markersize=8)
ax1.axhline(y=1, color='r', linestyle='--', label='Kaiser criterion (eigenvalue = 1)')
ax1.set_xlabel('Factor Number', fontsize=11, fontweight='bold')
ax1.set_ylabel('Eigenvalue', fontsize=11, fontweight='bold')
ax1.set_title('Scree Plot', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_xticks(range(1, min(9, len(eigenvalues) + 1)))

# Plot 2: Cumulative Variance Explained
ax2 = axes[0, 1]
cumvar = np.cumsum(eigenvalues) / np.sum(eigenvalues) * 100
ax2.plot(range(1, len(cumvar) + 1), cumvar, 'go-', linewidth=2, markersize=8)
ax2.axhline(y=80, color='r', linestyle='--', label='80% variance')
ax2.set_xlabel('Number of Factors', fontsize=11, fontweight='bold')
ax2.set_ylabel('Cumulative Variance Explained (%)', fontsize=11, fontweight='bold')
ax2.set_title('Cumulative Variance Explained', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_xticks(range(1, min(9, len(cumvar) + 1)))

# Plot 3: Factor Loadings Heatmap
ax3 = axes[1, 0]
sns.heatmap(loadings_df, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            cbar_kws={'label': 'Loading'}, ax=ax3, vmin=-1, vmax=1)
ax3.set_title('Factor Loadings Heatmap', fontsize=12, fontweight='bold')
ax3.set_xlabel('Factors', fontsize=11, fontweight='bold')
ax3.set_ylabel('Features', fontsize=11, fontweight='bold')

# Plot 4: Variance by Factor
ax4 = axes[1, 1]
explained_var_plot = explained_var / np.sum(explained_var) * 100
factor_labels = [factor_names.get(i+1, f"F{i+1}") for i in range(optimal_n_factors)]
bars = ax4.bar(range(optimal_n_factors), explained_var_plot, color='skyblue', edgecolor='navy', linewidth=1.5)
ax4.set_xlabel('Factor', fontsize=11, fontweight='bold')
ax4.set_ylabel('Variance Explained (%)', fontsize=11, fontweight='bold')
ax4.set_title('Variance Explained by Each Factor', fontsize=12, fontweight='bold')
ax4.set_xticks(range(optimal_n_factors))
ax4.set_xticklabels([f'F{i+1}' for i in range(optimal_n_factors)])
ax4.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, explained_var_plot)):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/factor_analysis_results.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved as 'factor_analysis_results.png'")
plt.show()

# ============================================================================
# 11. EXPORT RESULTS
# ============================================================================

# Save loadings to CSV
loadings_df.to_csv('/mnt/user-data/outputs/factor_loadings.csv')
print("✓ Factor loadings saved as 'factor_loadings.csv'")

# Save factor scores to CSV
factor_scores_df.to_csv('/mnt/user-data/outputs/factor_scores.csv')
print("✓ Factor scores saved as 'factor_scores.csv'")

# Create a summary report
summary_report = f"""
FACTOR ANALYSIS SUMMARY REPORT
{'=' * 80}

Dataset: Cat Breeds
Number of observations: {len(df_numeric)}
Number of original features: {len(numeric_cols)}
Number of factors extracted: {optimal_n_factors}

{'=' * 80}
FACTOR NAMES AND INTERPRETATION
{'=' * 80}

"""

for factor_idx in range(optimal_n_factors):
    factor_name = factor_names.get(factor_idx + 1, f"Factor {factor_idx + 1}")
    factor_var = explained_var[factor_idx] / np.sum(explained_var) * 100
    
    summary_report += f"\nFactor {factor_idx + 1}: {factor_name}\n"
    summary_report += f"Variance Explained: {factor_var:.2f}%\n"
    summary_report += "Top Contributing Features:\n"
    
    top_features_idx = np.argsort(np.abs(loadings[:, factor_idx]))[-3:][::-1]
    for i, idx in enumerate(top_features_idx, 1):
        feature_name = numeric_cols[idx]
        loading_value = loadings[idx, factor_idx]
        summary_report += f"  {i}. {feature_name}: {loading_value:.3f}\n"

summary_report += f"\n{'=' * 80}\n"
summary_report += f"Total Variance Explained: {total_variance*100:.2f}%\n"
summary_report += f"{'=' * 80}\n"

with open('/mnt/user-data/outputs/factor_analysis_report.txt', 'w') as f:
    f.write(summary_report)

print("✓ Summary report saved as 'factor_analysis_report.txt'")

print("\n" + "=" * 80)
print("FACTOR ANALYSIS COMPLETE")
print("=" * 80)
