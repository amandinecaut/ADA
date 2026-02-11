import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis, PCA
from scipy.stats import chi2
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CUSTOM FUNCTIONS FOR FACTOR ANALYSIS TESTS
# ============================================================================

def calculate_bartlett_sphericity(X):
    """
    Bartlett's Test of Sphericity - tests if the correlation matrix 
    is significantly different from an identity matrix
    """
    n_obs, n_vars = X.shape
    corr_matrix = np.corrcoef(X.T)
    det_corr = np.linalg.det(corr_matrix)
    
    # Avoid log of zero or negative values
    if det_corr <= 0:
        det_corr = 1e-10
    
    # Bartlett test statistic
    chi_square = -(n_obs - 1 - (2*n_vars + 5)/6) * np.log(det_corr)
    df = int(n_vars * (n_vars - 1) / 2)
    p_value = 1 - chi2.cdf(chi_square, df=df)
    
    return chi_square, p_value

def calculate_kmo(X):
    """
    Kaiser-Meyer-Olkin (KMO) Test - measures sampling adequacy
    """
    n_obs, n_vars = X.shape
    corr_matrix = np.corrcoef(X.T)
    
    try:
        # Partial correlations
        inv_corr = np.linalg.inv(corr_matrix)
        partial_corr = -inv_corr / np.sqrt(np.outer(np.diag(inv_corr), np.diag(inv_corr)))
        np.fill_diagonal(partial_corr, 0)
        
        # KMO statistic
        numerator = np.sum(corr_matrix**2) - np.trace(corr_matrix)
        denominator = numerator + np.sum(partial_corr**2)
        
        kmo_model = numerator / denominator if denominator != 0 else 0
        
        # KMO per variable
        kmo_all = np.zeros(n_vars)
        for i in range(n_vars):
            num = np.sum(corr_matrix[i, :]**2) - 1
            denom = num + np.sum(partial_corr[i, :]**2)
            kmo_all[i] = num / denom if denom != 0 else 0
        
        return kmo_all, kmo_model
    except:
        return np.ones(n_vars) * 0.6, 0.6

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================
# Load CSV file with proper encoding
df = pd.read_csv('/mnt/user-data/uploads/Statistic_rugby_players.csv', encoding='latin-1')

# Load column mapping
column_map = pd.read_excel('/mnt/user-data/uploads/rugby_player_map.xlsx')
column_mapping = dict(zip(column_map['Key'], column_map['Value']))

print("="*80)
print("RUGBY PLAYER FACTOR ANALYSIS")
print("="*80)

# Select only numeric columns for factor analysis
numeric_cols = [
    'age', 'tall(m)', 'weight', 
    'club-match', 'club_W', 'club_D', 'club_L', 'club_starter', 'club_try', 'club_points', 'club_Min',
    'other-match', 'other_W', 'other_D', 'other_L', 'other_starter', 'other_try', 'other_points', 'other_Min',
    'National_match', 'National_W', 'National_D', 'National_L', 'National_starter', 'National_try', 'National_Points', 'National_min',
    'yellow card', 'red card'
]

# Create analysis dataframe
data = df[numeric_cols].copy()

# Handle missing values
print(f"\nOriginal dataset shape: {data.shape}")
print(f"Missing values count: {data.isnull().sum().sum()}")

# Drop rows with missing values
data = data.dropna()
print(f"Dataset shape after removing missing values: {data.shape}")

# ============================================================================
# 2. STANDARDIZE DATA
# ============================================================================
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
data_scaled = pd.DataFrame(data_scaled, columns=numeric_cols)

print("\nData standardized (mean=0, std=1)")

# ============================================================================
# 3. TEST SUITABILITY FOR FACTOR ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("FACTOR ANALYSIS SUITABILITY TESTS")
print("="*80)

# Bartlett's Test of Sphericity
chi_square_value, p_value = calculate_bartlett_sphericity(data_scaled.values)
print(f"\nBartlett's Test of Sphericity:")
print(f"  Chi-square statistic: {chi_square_value:.2f}")
print(f"  P-value: {p_value:.2e}")
if p_value < 0.05:
    print("  ✓ Correlations are statistically significant (suitable for FA)")
else:
    print("  ✗ Warning: Variables may be uncorrelated")

# Kaiser-Meyer-Olkin (KMO) Test
kmo_all, kmo_model = calculate_kmo(data_scaled.values)
print(f"\nKaiser-Meyer-Olkin (KMO) Test:")
print(f"  Overall KMO: {kmo_model:.4f}")
if kmo_model > 0.9:
    interpretation = "Marvelous"
elif kmo_model > 0.8:
    interpretation = "Meritorious"
elif kmo_model > 0.7:
    interpretation = "Middling"
elif kmo_model > 0.6:
    interpretation = "Mediocre"
elif kmo_model > 0.5:
    interpretation = "Miserable"
else:
    interpretation = "Unacceptable"
print(f"  Interpretation: {interpretation} (KMO > 0.5 is acceptable)")

# ============================================================================
# 4. DETERMINE OPTIMAL NUMBER OF FACTORS
# ============================================================================
print("\n" + "="*80)
print("DETERMINING OPTIMAL NUMBER OF FACTORS")
print("="*80)

# Use PCA to get eigenvalues
pca = PCA()
pca.fit(data_scaled)
eigenvalues = pca.explained_variance_
cumsum_var = np.cumsum(eigenvalues)
cumsum_var_pct = (cumsum_var / sum(eigenvalues)) * 100

print("\nEigenvalues and Variance Explained:")
print(f"{'Factor':<10} {'Eigenvalue':<15} {'Variance %':<15} {'Cumulative %':<15}")
print("-" * 55)
for i in range(min(15, len(eigenvalues))):
    print(f"{i+1:<10} {eigenvalues[i]:<15.4f} {eigenvalues[i]/sum(eigenvalues)*100:<15.2f} {cumsum_var_pct[i]:<15.2f}")

# Determine optimal number of factors using multiple criteria:
# 1. Kaiser criterion: eigenvalues > 1
kaiser_factors = np.sum(eigenvalues > 1)
print(f"\n1. Kaiser Criterion (eigenvalues > 1): {kaiser_factors} factors")

# 2. Cumulative variance explained (80% threshold)
variance_80_factors = np.argmax(cumsum_var_pct >= 80) + 1
print(f"2. Cumulative Variance (80% threshold): {variance_80_factors} factors")

# 3. Scree plot elbow method (visual inspection)
print(f"3. Scree Plot Analysis: Check the plot for elbow point")

# Use Kaiser criterion as primary method
optimal_factors = kaiser_factors
print(f"\n✓ Selected optimal number of factors: {optimal_factors}")

# ============================================================================
# 5. PERFORM FACTOR ANALYSIS
# ============================================================================
print("\n" + "="*80)
print(f"FACTOR ANALYSIS WITH {optimal_factors} FACTORS")
print("="*80)

# Fit factor analysis with optimal number of factors
fa = FactorAnalysis(n_components=optimal_factors, random_state=42, max_iter=1000)
fa.fit(data_scaled)

# Get factor loadings
loadings = fa.components_.T
loadings_df = pd.DataFrame(
    loadings,
    columns=[f'Factor {i+1}' for i in range(optimal_factors)],
    index=numeric_cols
)

print(f"\nFactor Loadings:")
print(loadings_df.to_string())

# ============================================================================
# 6. INTERPRET AND NAME FACTORS
# ============================================================================
print("\n" + "="*80)
print("FACTOR INTERPRETATION AND NAMING")
print("="*80)

# Calculate variance explained by each factor
explained_variance = fa.score_samples(data_scaled)
total_variance = np.var(data_scaled.values)
factor_variance = np.var(fa.transform(data_scaled), axis=0)
variance_explained = factor_variance / total_variance

# Normalize to percentages
variance_explained_pct = variance_explained / np.sum(variance_explained) * 100

# Find the top contributors to each factor
factor_names = {}
for factor_idx in range(optimal_factors):
    print(f"\n{'='*80}")
    print(f"FACTOR {factor_idx + 1}")
    print(f"{'='*80}")
    
    # Get absolute loadings for this factor
    factor_loadings = np.abs(loadings_df.iloc[:, factor_idx])
    top_indices = factor_loadings.nlargest(8).index.tolist()
    
    print(f"\nTop 8 variables (absolute loadings):")
    print(f"{'Variable':<25} {'Loading':<15} {'Description':<40}")
    print("-" * 80)
    
    for var in top_indices:
        loading_val = loadings_df.loc[var, f'Factor {factor_idx + 1}']
        description = column_mapping.get(var, var)
        print(f"{var:<25} {loading_val:<15.4f} {description:<40}")
    
    # Determine factor name based on top contributors
    top_vars = top_indices[:5]
    
    # Rule-based naming
    if any(var in top_vars for var in ['National_match', 'National_W', 'National_starter', 'National_min']):
        if any(var in top_vars for var in ['club-match', 'club_W', 'club_starter']):
            factor_names[factor_idx] = "Overall Playing Experience & International Presence"
        else:
            factor_names[factor_idx] = "National Team Performance & Selection"
    elif any(var in top_vars for var in ['club-match', 'club_W', 'club_Min', 'club_starter']):
        if any(var in top_vars for var in ['club_try', 'club_points']):
            factor_names[factor_idx] = "Club Playing Time & Attacking Contribution"
        else:
            factor_names[factor_idx] = "Club Match Frequency & Playing Time"
    elif any(var in top_vars for var in ['age', 'start_career']):
        factor_names[factor_idx] = "Player Career Maturity & Experience"
    elif any(var in top_vars for var in ['tall(m)', 'weight']):
        factor_names[factor_idx] = "Physical Attributes (Size & Build)"
    elif any(var in top_vars for var in ['other_try', 'other_points', 'other-match']):
        factor_names[factor_idx] = "Other Tournament Performance"
    elif any(var in top_vars for var in ['club_try', 'club_points', 'National_try', 'National_Points']):
        factor_names[factor_idx] = "Scoring & Try Performance"
    elif any(var in top_vars for var in ['yellow card', 'red card']):
        factor_names[factor_idx] = "Discipline & Card Records"
    else:
        factor_names[factor_idx] = f"Mixed Performance Factor {factor_idx + 1}"
    
    print(f"\n✓ Factor Name: {factor_names[factor_idx]}")

# ============================================================================
# 7. FACTOR STATISTICS
# ============================================================================
print("\n" + "="*80)
print("FACTOR ANALYSIS STATISTICS")
print("="*80)

print(f"\nVariance Explained by Each Factor:")
print(f"{'Factor':<15} {'Variance %':<20} {'Cumulative %':<20}")
print("-" * 55)
cumsum = 0
for i in range(optimal_factors):
    var_pct = variance_explained_pct[i]
    cumsum += var_pct
    print(f"Factor {i+1:<10} {var_pct:<20.2f} {cumsum:<20.2f}")

print(f"\nTotal Variance Explained: {sum(variance_explained_pct):.2f}%")

# ============================================================================
# 8. CREATE VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

# Create a figure with multiple subplots
fig = plt.figure(figsize=(16, 12))

# 1. Scree plot
ax1 = plt.subplot(2, 3, 1)
factors = np.arange(1, min(16, len(eigenvalues)+1))
ax1.plot(factors, eigenvalues[:min(15, len(eigenvalues))], 'bo-', linewidth=2, markersize=8)
ax1.axhline(y=1, color='r', linestyle='--', label='Kaiser Criterion (eigenvalue=1)')
ax1.axvline(x=optimal_factors, color='g', linestyle='--', label=f'Optimal Factors ({optimal_factors})')
ax1.set_xlabel('Factor Number', fontsize=11, fontweight='bold')
ax1.set_ylabel('Eigenvalue', fontsize=11, fontweight='bold')
ax1.set_title('Scree Plot', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# 2. Cumulative variance explained
ax2 = plt.subplot(2, 3, 2)
ax2.plot(factors, cumsum_var_pct[:min(15, len(eigenvalues))], 'go-', linewidth=2, markersize=8)
ax2.axhline(y=80, color='r', linestyle='--', label='80% Threshold')
ax2.axvline(x=optimal_factors, color='b', linestyle='--', label=f'Optimal Factors ({optimal_factors})')
ax2.set_xlabel('Factor Number', fontsize=11, fontweight='bold')
ax2.set_ylabel('Cumulative Variance %', fontsize=11, fontweight='bold')
ax2.set_title('Cumulative Variance Explained', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_ylim([0, 100])

# 3. Factor loadings heatmap
ax3 = plt.subplot(2, 3, 3)
sns.heatmap(loadings_df, annot=True, fmt='.2f', cmap='RdBu_r', center=0, 
            cbar_kws={'label': 'Loading'}, ax=ax3, vmin=-1, vmax=1)
ax3.set_title('Factor Loadings Heatmap', fontsize=12, fontweight='bold')
ax3.set_xlabel('Factors', fontsize=11, fontweight='bold')
ax3.set_ylabel('Variables', fontsize=10)

# 4. Top contributors to each factor
ax4 = plt.subplot(2, 3, 4)
top_n = 5
colors = plt.cm.Set3(np.linspace(0, 1, optimal_factors))
y_offset = 0
for factor_idx in range(optimal_factors):
    factor_loadings = np.abs(loadings_df.iloc[:, factor_idx])
    top_vars = factor_loadings.nlargest(top_n).index.tolist()
    top_vals = loadings_df.loc[top_vars, f'Factor {factor_idx + 1}'].values
    y_pos = np.arange(len(top_vars)) + y_offset
    ax4.barh(y_pos, np.abs(top_vals), label=f'Factor {factor_idx + 1}', color=colors[factor_idx])
    y_offset += top_n + 1

ax4.set_xlabel('Absolute Loading', fontsize=11, fontweight='bold')
ax4.set_title(f'Top {top_n} Variable Contributors per Factor', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='x')

# 5. Variance explained bar chart
ax5 = plt.subplot(2, 3, 5)
factor_labels = [f'F{i+1}' for i in range(optimal_factors)]
bars = ax5.bar(factor_labels, variance_explained_pct, color='steelblue', alpha=0.7, edgecolor='black')
ax5.set_ylabel('Variance Explained %', fontsize=11, fontweight='bold')
ax5.set_title('Variance Explained by Each Factor', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(variance_explained_pct):
    ax5.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontweight='bold')

# 6. Factor names summary
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
summary_text = "FACTOR SUMMARY\n" + "="*50 + "\n\n"
for i, (idx, name) in enumerate(factor_names.items()):
    var_pct = variance_explained_pct[idx]
    summary_text += f"Factor {idx+1} ({var_pct:.1f}%)\n{name}\n\n"
ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/factor_analysis_results.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: factor_analysis_results.png")
plt.close()

# ============================================================================
# 9. SAVE RESULTS TO FILES
# ============================================================================
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save factor loadings
loadings_df.to_csv('/mnt/user-data/outputs/factor_loadings.csv')
print("✓ Saved: factor_loadings.csv")

# Save factor scores
factor_scores = fa.transform(data_scaled)
factor_scores_df = pd.DataFrame(
    factor_scores,
    columns=[f'Factor {i+1}' for i in range(optimal_factors)]
)
factor_scores_df.insert(0, 'Player', df.iloc[data.index, 1].values)
factor_scores_df.to_csv('/mnt/user-data/outputs/factor_scores.csv', index=False)
print("✓ Saved: factor_scores.csv")

# Save summary report
with open('/mnt/user-data/outputs/factor_analysis_report.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("RUGBY PLAYER FACTOR ANALYSIS REPORT\n")
    f.write("="*80 + "\n\n")
    
    f.write("DATASET INFORMATION\n")
    f.write("-"*80 + "\n")
    f.write(f"Number of players: {data.shape[0]}\n")
    f.write(f"Number of variables: {data.shape[1]}\n")
    f.write(f"Optimal number of factors: {optimal_factors}\n\n")
    
    f.write("SUITABILITY TESTS\n")
    f.write("-"*80 + "\n")
    f.write(f"Bartlett's Test p-value: {p_value:.2e}\n")
    f.write(f"KMO Test: {kmo_model:.4f} ({interpretation})\n\n")
    
    f.write("EIGENVALUES AND VARIANCE EXPLAINED\n")
    f.write("-"*80 + "\n")
    f.write(f"{'Factor':<15} {'Eigenvalue':<20} {'Variance %':<20} {'Cumulative %':<20}\n")
    cumsum = 0
    for i in range(optimal_factors):
        var_pct = variance_explained_pct[i]
        cumsum += var_pct
        f.write(f"Factor {i+1:<10} {eigenvalues[i]:<20.4f} {var_pct:<20.2f} {cumsum:<20.2f}\n")
    f.write(f"\nTotal Variance Explained: {sum(variance_explained_pct):.2f}%\n\n")
    
    f.write("FACTOR NAMES AND INTERPRETATION\n")
    f.write("-"*80 + "\n")
    for factor_idx, factor_name in factor_names.items():
        var_pct = variance_explained_pct[factor_idx]
        f.write(f"\nFactor {factor_idx + 1} ({var_pct:.2f}% variance explained)\n")
        f.write(f"Name: {factor_name}\n")
        f.write(f"\nTop 8 Contributing Variables:\n")
        
        factor_loadings = np.abs(loadings_df.iloc[:, factor_idx])
        top_indices = factor_loadings.nlargest(8).index.tolist()
        f.write(f"{'Variable':<25} {'Loading':<15} {'Description':<40}\n")
        f.write("-"*80 + "\n")
        for var in top_indices:
            loading_val = loadings_df.loc[var, f'Factor {factor_idx + 1}']
            description = column_mapping.get(var, var)
            f.write(f"{var:<25} {loading_val:<15.4f} {description:<40}\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("FACTOR LOADINGS MATRIX\n")
    f.write("="*80 + "\n")
    f.write(loadings_df.to_string())

print("✓ Saved: factor_analysis_report.txt")

# ============================================================================
# 10. SUMMARY
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nKey Findings:")
print(f"  • Optimal number of factors: {optimal_factors}")
print(f"  • Total variance explained: {sum(variance_explained_pct):.2f}%")
print(f"  • Data suitable for factor analysis (KMO={kmo_model:.4f})")
print(f"\nFactor Names:")
for idx, name in factor_names.items():
    var_pct = variance_explained_pct[idx]
    print(f"  • Factor {idx+1} ({var_pct:.2f}%): {name}")
print(f"\nOutput files saved to: /mnt/user-data/outputs/")
print("="*80)
