import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis, PCA
from scipy.linalg import inv
import warnings
warnings.filterwarnings('ignore')

# Custom implementation of Bartlett's test
def calculate_bartlett_sphericity(X):
    """Calculate Bartlett's test of sphericity"""
    n, p = X.shape
    corr_matrix = np.corrcoef(X.T)
    chi_square = -(n - 1 - (2*p + 5)/6) * np.log(np.linalg.det(corr_matrix))
    df = p * (p - 1) / 2
    # Approximate p-value using chi-square approximation
    from scipy.stats import chi2
    p_value = 1 - chi2.cdf(chi_square, df)
    return chi_square, p_value

# Custom implementation of KMO test
def calculate_kmo(X):
    """Calculate Kaiser-Meyer-Olkin (KMO) test"""
    corr_matrix = np.corrcoef(X.T)
    
    # Get partial correlation matrix
    try:
        partial_corr = -inv(corr_matrix)
        # Normalize by diagonal
        diag = np.sqrt(np.diag(partial_corr))
        partial_corr_norm = partial_corr / np.outer(diag, diag)
    except:
        partial_corr_norm = np.zeros_like(corr_matrix)
    
    np.fill_diagonal(partial_corr_norm, 0)
    
    corr_norm = corr_matrix.copy()
    np.fill_diagonal(corr_norm, 0)
    
    kmo_numerator = np.sum(corr_norm ** 2)
    kmo_denominator = np.sum(corr_norm ** 2) + np.sum(partial_corr_norm ** 2)
    
    kmo_overall = kmo_numerator / kmo_denominator if kmo_denominator > 0 else 0
    kmo_per_variable = np.array([
        np.sum(corr_norm[i] ** 2) / (np.sum(corr_norm[i] ** 2) + np.sum(partial_corr_norm[i] ** 2))
        if (np.sum(corr_norm[i] ** 2) + np.sum(partial_corr_norm[i] ** 2)) > 0 else 0
        for i in range(len(corr_matrix))
    ])
    
    return kmo_per_variable, kmo_overall

# ==================== Load Data ====================
csv_file = '/mnt/user-data/uploads/team_stats.csv'
json_file = '/mnt/user-data/uploads/match_api_metric_map.json'

# Load CSV
df = pd.read_csv(csv_file)
print("Data shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# Load JSON mapping
with open(json_file, 'r') as f:
    metric_map = json.load(f)

print(f"\nTotal columns: {len(df.columns)}")

# ==================== Data Preparation ====================
# Select only numeric columns for analysis (exclude identifiers)
exclude_cols = ['team_id', 'club_name', 'competition_id', 'season']
numeric_cols = [col for col in df.columns if col not in exclude_cols]

# Create subset for analysis
X = df[numeric_cols].copy()

# Handle missing values
X = X.dropna()
print(f"\nData after removing missing values: {X.shape}")

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=numeric_cols)

print(f"Standardized data shape: {X_scaled.shape}")

# ==================== Test for Factorability ====================
print("\n" + "="*60)
print("FACTORABILITY TESTS")
print("="*60)

# Bartlett's Test
chi_square_value, p_value = calculate_bartlett_sphericity(X_scaled.values)
print(f"\nBartlett's Test of Sphericity:")
print(f"  Chi-square: {chi_square_value:.2f}")
print(f"  P-value: {p_value:.2e}")
print(f"  Result: {'Data is suitable for FA' if p_value < 0.05 else 'Data may not be suitable for FA'}")

# KMO Test
kmo_all, kmo_model = calculate_kmo(X_scaled.values)
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

# ==================== Determine Optimal Number of Factors ====================
print("\n" + "="*60)
print("DETERMINING OPTIMAL NUMBER OF FACTORS")
print("="*60)

# Use PCA to get eigenvalues
pca = PCA()
pca.fit(X_scaled)
eigenvalues = pca.explained_variance_

print(f"\nEigenvalues (first 20):")
for i, ev in enumerate(eigenvalues[:min(20, len(eigenvalues))], 1):
    print(f"  Factor {i}: {ev:.4f}")

# Plot scree plot
plt.figure(figsize=(12, 5))

# Scree plot
plt.subplot(1, 2, 1)
plt.plot(range(1, min(21, len(eigenvalues)+1)), eigenvalues[:20], 'bo-', linewidth=2, markersize=8)
plt.axhline(y=1, color='r', linestyle='--', label='Kaiser Criterion (eigenvalue=1)')
plt.xlabel('Factor Number', fontsize=12)
plt.ylabel('Eigenvalue', fontsize=12)
plt.title('Scree Plot', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()

# Cumulative variance explained
cumsum = np.cumsum(pca.explained_variance_ratio_)
plt.subplot(1, 2, 2)
plt.plot(range(1, min(21, len(eigenvalues)+1)), cumsum[:20]*100, 'go-', linewidth=2, markersize=8)
plt.axhline(y=80, color='r', linestyle='--', label='80% Variance')
plt.axhline(y=90, color='orange', linestyle='--', label='90% Variance')
plt.xlabel('Number of Factors', fontsize=12)
plt.ylabel('Cumulative Variance Explained (%)', fontsize=12)
plt.title('Cumulative Variance Explained', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/scree_plot.png', dpi=300, bbox_inches='tight')
print("\nScree plot saved!")

# Determine optimal factors using Kaiser criterion (eigenvalue > 1)
n_factors_kaiser = sum(eigenvalues > 1)
print(f"\nOptimal factors (Kaiser criterion - eigenvalue > 1): {n_factors_kaiser}")

# Also check for 80% and 90% variance
n_factors_80 = sum(cumsum < 0.80) + 1
n_factors_90 = sum(cumsum < 0.90) + 1
print(f"Factors for 80% variance explained: {n_factors_80}")
print(f"Factors for 90% variance explained: {n_factors_90}")
if n_factors_kaiser > 0:
    print(f"Cumulative variance with {n_factors_kaiser} factors: {cumsum[n_factors_kaiser-1]*100:.2f}%")

# Use Kaiser criterion as optimal, but ensure at least 2 factors
optimal_n_factors = max(2, n_factors_kaiser)
print(f"\n>>> Using {optimal_n_factors} factors for analysis <<<")

# ==================== Perform Factor Analysis ====================
print("\n" + "="*60)
print(f"FACTOR ANALYSIS WITH {optimal_n_factors} FACTORS")
print("="*60)

# Fit FactorAnalysis model
fa = FactorAnalysis(n_components=optimal_n_factors, max_iter=500, random_state=42)
fa.fit(X_scaled)

# Get factor loadings
loadings = fa.components_.T
loadings_df = pd.DataFrame(
    loadings,
    index=numeric_cols,
    columns=[f'Factor {i+1}' for i in range(optimal_n_factors)]
)

print("\nFactor Loadings:")
print(loadings_df.round(3))

# Calculate variance explained by each factor
# Variance = sum of squared loadings / number of variables
variance_per_factor = np.sum(loadings**2, axis=0) / len(numeric_cols)
total_variance = np.sum(variance_per_factor)
variance_pct = (variance_per_factor / total_variance * 100) if total_variance > 0 else np.zeros(len(variance_per_factor))

variance_df = pd.DataFrame({
    'Variance': variance_per_factor,
    'Cumulative Variance': np.cumsum(variance_per_factor),
    'Variance %': variance_pct,
    'Cumulative %': np.cumsum(variance_pct)
}, index=[f'Factor {i+1}' for i in range(optimal_n_factors)])

print("\nVariance Explained:")
print(variance_df.round(4))

# ==================== Name Factors ====================
print("\n" + "="*60)
print("FACTOR NAMING AND INTERPRETATION")
print("="*60)

# Get the top contributors to each factor
factor_names = {}
factor_interpretations = {}

for factor_idx in range(optimal_n_factors):
    loadings_col = loadings_df.iloc[:, factor_idx].abs().sort_values(ascending=False)
    loadings_col_signed = loadings_df.iloc[:, factor_idx].sort_values(ascending=False)
    
    top_positive = loadings_col_signed.head(5)
    top_negative = loadings_col_signed.tail(5)[::-1]
    
    print(f"\n{'='*60}")
    print(f"FACTOR {factor_idx + 1}")
    print(f"Variance Explained: {variance_df.iloc[factor_idx, 3]:.2f}%")
    print(f"{'='*60}")
    
    print(f"\nTop Positive Loadings (Strength indicators):")
    for var, loading in top_positive.items():
        full_name = metric_map.get(var, var)
        print(f"  {full_name}: {loading:.4f}")
    
    print(f"\nTop Negative Loadings (Weakness indicators):")
    for var, loading in top_negative.items():
        full_name = metric_map.get(var, var)
        print(f"  {full_name}: {loading:.4f}")
    
    # Automated naming based on variable characteristics
    top_vars = list(top_positive.index[:5])
    top_var_names = [metric_map.get(v, v).lower() for v in top_vars]
    
    # Simple heuristics for naming
    name = "General Performance"
    
    if any('box' in v for v in top_var_names):
        if any('touch' in v for v in top_var_names):
            name = "Box Dominance"
        else:
            name = "Attacking Efficiency"
    elif any('possession' in v for v in top_var_names):
        name = "Possession Control"
    elif any('defensive' in v for v in top_var_names):
        name = "Defensive Prowess"
    elif any('recovery' in v for v in top_var_names):
        name = "Recovery & Transition"
    elif any('pass' in v for v in top_var_names):
        name = "Passing Quality"
    elif any('goal' in v for v in top_var_names) or any('xg' in v for v in top_var_names):
        name = "Attacking Output"
    elif any('foul' in v for v in top_var_names) or any('card' in v for v in top_var_names):
        name = "Discipline & Aggression"
    elif any('xt' in v for v in top_var_names):
        name = "Threat Creation"
    
    factor_names[f"Factor {factor_idx + 1}"] = name
    factor_interpretations[f"Factor {factor_idx + 1}"] = {
        'name': name,
        'variance_explained': float(variance_df.iloc[factor_idx, 3]),
        'top_positive_indicators': list(top_positive.index[:5]),
        'top_negative_indicators': list(top_negative.index[:5])
    }
    
    print(f"\n>>> Suggested Factor Name: {name} <<<")

# ==================== Visualization ====================
print("\n" + "="*60)
print("CREATING VISUALIZATIONS")
print("="*60)

# Plot loadings heatmap
plt.figure(figsize=(10, max(8, len(numeric_cols)*0.25)))
sns.heatmap(loadings_df, cmap='RdBu_r', center=0, annot=True, fmt='.2f', 
            cbar_kws={'label': 'Loading'}, linewidths=0.5)
plt.title('Factor Loadings Heatmap', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Factors', fontsize=12)
plt.ylabel('Variables', fontsize=12)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/factor_loadings_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Factor loadings heatmap saved!")

# Create bar plots for top loadings in each factor
n_top = 12
fig, axes = plt.subplots(1, optimal_n_factors, figsize=(6*optimal_n_factors, 8))
if optimal_n_factors == 1:
    axes = [axes]

for factor_idx in range(optimal_n_factors):
    loadings_col = loadings_df.iloc[:, factor_idx].sort_values()
    
    # Get top n positive and negative
    top_n_pos = loadings_col.tail(n_top//2)
    top_n_neg = loadings_col.head(n_top//2)
    top_combined = pd.concat([top_n_neg, top_n_pos])
    
    colors = ['#d62728' if x < 0 else '#2ca02c' for x in top_combined.values]
    
    axes[factor_idx].barh(range(len(top_combined)), top_combined.values, color=colors, alpha=0.8)
    axes[factor_idx].set_yticks(range(len(top_combined)))
    axes[factor_idx].set_yticklabels([metric_map.get(v, v)[:40] for v in top_combined.index], fontsize=9)
    axes[factor_idx].set_xlabel('Loading', fontsize=11)
    axes[factor_idx].set_title(f'{factor_names[f"Factor {factor_idx + 1}"]}\n({variance_df.iloc[factor_idx, 3]:.1f}% var)', 
                               fontsize=12, fontweight='bold')
    axes[factor_idx].axvline(x=0, color='black', linewidth=0.8)
    axes[factor_idx].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/top_loadings_by_factor.png', dpi=300, bbox_inches='tight')
print("✓ Top loadings visualization saved!")

# Variance explained pie chart
plt.figure(figsize=(8, 6))
colors = plt.cm.Set3(np.linspace(0, 1, optimal_n_factors))
plt.pie(variance_df['Variance %'], labels=[factor_names[f"Factor {i+1}"] for i in range(optimal_n_factors)],
        autopct='%1.1f%%', colors=colors, startangle=90)
plt.title('Variance Explained by Factor', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/variance_pie_chart.png', dpi=300, bbox_inches='tight')
print("✓ Variance pie chart saved!")

# ==================== Save Results ====================
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

# Save detailed loadings
loadings_df.to_csv('/mnt/user-data/outputs/factor_loadings.csv')
print("✓ Factor loadings saved to factor_loadings.csv")

# Save variance explained
variance_df.to_csv('/mnt/user-data/outputs/variance_explained.csv')
print("✓ Variance explained saved to variance_explained.csv")

# Save factor scores
factor_scores = fa.transform(X_scaled)
factor_scores_df = pd.DataFrame(
    factor_scores,
    columns=[f'{factor_names[f"Factor {i+1}"]}'.replace(' ', '_') for i in range(optimal_n_factors)]
)
factor_scores_df.to_csv('/mnt/user-data/outputs/factor_scores.csv', index=False)
print("✓ Factor scores saved to factor_scores.csv")

# Save factor interpretations
with open('/mnt/user-data/outputs/factor_interpretations.json', 'w') as f:
    json.dump(factor_interpretations, f, indent=2)
print("✓ Factor interpretations saved to factor_interpretations.json")

# Save summary report
with open('/mnt/user-data/outputs/factor_analysis_summary.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("FACTOR ANALYSIS SUMMARY REPORT\n")
    f.write("="*70 + "\n\n")
    
    f.write(f"Data: {len(df)} teams, {len(numeric_cols)} variables\n")
    f.write(f"After cleaning: {len(X_scaled)} observations\n\n")
    
    f.write("FACTORABILITY TESTS\n")
    f.write("-"*70 + "\n")
    f.write(f"Bartlett's Test Chi-square: {chi_square_value:.2f}\n")
    f.write(f"Bartlett's Test p-value: {p_value:.2e}\n")
    f.write(f"Suitable for Factor Analysis: {p_value < 0.05}\n")
    f.write(f"KMO Index: {kmo_model:.4f}\n\n")
    
    f.write("OPTIMAL NUMBER OF FACTORS\n")
    f.write("-"*70 + "\n")
    f.write(f"Kaiser Criterion (eigenvalue > 1): {n_factors_kaiser}\n")
    f.write(f"Factors for 80% variance: {n_factors_80}\n")
    f.write(f"Factors for 90% variance: {n_factors_90}\n")
    f.write(f"Selected: {optimal_n_factors} factors\n")
    f.write(f"Total variance explained: {cumsum[optimal_n_factors-1]*100:.2f}%\n\n")
    
    f.write("FACTOR NAMES AND INTERPRETATIONS\n")
    f.write("-"*70 + "\n")
    for i in range(optimal_n_factors):
        factor_name = factor_names[f"Factor {i+1}"]
        variance_pct = variance_df.iloc[i, 3]
        f.write(f"\nFactor {i+1}: {factor_name}\n")
        f.write(f"  Variance Explained: {variance_pct:.2f}%\n")
        f.write(f"  Top Positive Indicators:\n")
        for var in factor_interpretations[f"Factor {i+1}"]['top_positive_indicators'][:3]:
            loading = loadings_df.loc[var, f"Factor {i+1}"]
            f.write(f"    - {metric_map.get(var, var)}: {loading:.4f}\n")
        f.write(f"  Top Negative Indicators:\n")
        for var in factor_interpretations[f"Factor {i+1}"]['top_negative_indicators'][:3]:
            loading = loadings_df.loc[var, f"Factor {i+1}"]
            f.write(f"    - {metric_map.get(var, var)}: {loading:.4f}\n")

print("✓ Summary report saved to factor_analysis_summary.txt")

print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)
print("\nOutput files created:")
print("  1. factor_loadings.csv - Detailed factor loadings")
print("  2. variance_explained.csv - Variance explained by each factor")
print("  3. factor_scores.csv - Factor scores for each team")
print("  4. factor_interpretations.json - JSON with factor details")
print("  5. factor_analysis_summary.txt - Text summary report")
print("  6. scree_plot.png - Eigenvalue and variance explained plots")
print("  7. factor_loadings_heatmap.png - Heatmap of all loadings")
print("  8. top_loadings_by_factor.png - Bar charts of top loadings")
print("  9. variance_pie_chart.png - Pie chart of variance by factor")
print("\n" + "="*70)
