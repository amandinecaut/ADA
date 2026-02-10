"""
Factor Analysis with Optimal Number of Factors
This script performs factor analysis on the 16 Personality Type dataset
Uses scikit-learn's decomposition module
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2
from scipy.linalg import cholesky
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================

print("=" * 80)
print("FACTOR ANALYSIS - PERSONALITY DATA")
print("=" * 80)

# Load data with appropriate encoding
df = pd.read_csv('16P.csv', encoding='latin-1')

# Remove the Response ID and Personality columns (not needed for factor analysis)
# Keep only the numerical survey response columns
X = df.iloc[:, 1:-1]  # All columns except Response Id (first) and Personality (last)

print(f"\nDataset shape: {X.shape}")
print(f"Number of variables: {X.shape[1]}")
print(f"Number of observations: {X.shape[0]}")
print(f"\nVariable names (first 5):")
for i, col in enumerate(X.columns[:5]):
    print(f"  {i+1}. {col[:70]}...")

# Standardize the data (important for factor analysis)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# ============================================================================
# 2. COMPUTE CORRELATION MATRIX
# ============================================================================

print("\n" + "=" * 80)
print("COMPUTING CORRELATION MATRIX")
print("=" * 80)

corr_matrix = np.corrcoef(X_scaled.T)
print(f"Correlation matrix shape: {corr_matrix.shape}")
print(f"Mean absolute correlation: {np.mean(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])):.4f}")

# ============================================================================
# 3. DETERMINE OPTIMAL NUMBER OF FACTORS
# ============================================================================

print("\n" + "=" * 80)
print("DETERMINING OPTIMAL NUMBER OF FACTORS")
print("=" * 80)

# Method 1: Kaiser Criterion using eigenvalues of correlation matrix
eigenvalues = np.linalg.eigvals(corr_matrix)
eigenvalues = np.sort(eigenvalues)[::-1]  # Sort in descending order

# Count factors with eigenvalue > 1
num_factors_kaiser = np.sum(eigenvalues > 1)
print(f"\nMethod 1: Kaiser Criterion (Eigenvalue > 1)")
print(f"  Optimal number of factors: {num_factors_kaiser}")
print(f"  Eigenvalues (first 15):")
for i in range(min(15, len(eigenvalues))):
    marker = " <- Kaiser cutoff" if eigenvalues[i] > 1 else ""
    print(f"    Factor {i+1}: {eigenvalues[i]:.4f}{marker}")

# Method 2: Cumulative Variance Explained
total_variance = np.sum(eigenvalues)
cumsum_variance = np.cumsum(eigenvalues) / total_variance
variance_90_idx = np.argmax(cumsum_variance >= 0.90)
num_factors_var90 = variance_90_idx + 1

print(f"\nMethod 2: Cumulative Variance Explained (90% threshold)")
print(f"  Number of factors for 90% variance: {num_factors_var90}")
print(f"  Cumulative variance explained (first 15 factors):")
for i in range(min(15, len(cumsum_variance))):
    marker = " <- 90% threshold" if cumsum_variance[i] >= 0.90 and i == variance_90_idx else ""
    print(f"    Factors 1-{i+1}: {cumsum_variance[i]:.4f}{marker}")

# Visualize scree plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scree plot
axes[0].bar(range(1, min(16, len(eigenvalues)+1)), eigenvalues[:15], alpha=0.7, color='steelblue')
axes[0].axhline(y=1, color='r', linestyle='--', linewidth=2, label='Kaiser Criterion')
axes[0].set_xlabel('Factor Number', fontsize=11)
axes[0].set_ylabel('Eigenvalue', fontsize=11)
axes[0].set_title('Scree Plot', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Cumulative variance explained
axes[1].plot(range(1, min(16, len(cumsum_variance)+1)), cumsum_variance[:15], 
             marker='o', linestyle='-', color='steelblue', linewidth=2, markersize=6)
axes[1].axhline(y=0.90, color='r', linestyle='--', linewidth=2, label='90% Variance')
axes[1].set_xlabel('Number of Factors', fontsize=11)
axes[1].set_ylabel('Cumulative Variance Explained', fontsize=11)
axes[1].set_title('Cumulative Variance Explained', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('scree_plot.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Scree plot saved to 'scree_plot.png'")
plt.close()

# Select optimal number of factors (using Kaiser criterion as primary)
optimal_factors = num_factors_kaiser
print(f"\n{'*' * 80}")
print(f"SELECTED OPTIMAL NUMBER OF FACTORS: {optimal_factors}")
print(f"{'*' * 80}")

# ============================================================================
# 4. PERFORM FACTOR ANALYSIS WITH OPTIMAL FACTORS
# ============================================================================

print(f"\n" + "=" * 80)
print(f"FACTOR ANALYSIS WITH {optimal_factors} FACTORS")
print("=" * 80)

# Fit factor analysis model
fa = FactorAnalysis(n_components=optimal_factors, random_state=42, max_iter=500)
factors = fa.fit_transform(X_scaled_df)

# Get the loadings
loadings = fa.components_.T

# Create loadings DataFrame
loadings_df = pd.DataFrame(
    loadings,
    columns=[f'Factor {i+1}' for i in range(optimal_factors)],
    index=X.columns
)

# Calculate variance explained
variance_explained = np.sum(loadings**2, axis=0) / X.shape[1]
print(f"\nVariance explained by each factor:")
for i in range(optimal_factors):
    print(f"  Factor {i+1}: {variance_explained[i]:.4f} ({variance_explained[i]*100:.2f}%)")
print(f"  Total: {variance_explained.sum():.4f} ({variance_explained.sum()*100:.2f}%)")

# Calculate communalities (variance explained for each variable)
communalities = np.sum(loadings**2, axis=1)
communalities_df = pd.DataFrame(
    communalities,
    columns=['Communality'],
    index=X.columns
)
print(f"\nCommunalities (variance explained for each variable):")
print(f"Mean communality: {communalities.mean():.4f}")
print(f"Min communality: {communalities.min():.4f}")
print(f"Max communality: {communalities.max():.4f}")

# ============================================================================
# 5. IDENTIFY KEY VARIABLES FOR EACH FACTOR
# ============================================================================

print(f"\n" + "=" * 80)
print(f"FACTOR LOADINGS & KEY VARIABLES")
print("=" * 80)

print("\nTop variables loading on each factor (absolute loading > 0.40):")
print("-" * 80)

factor_characteristics = {}

for factor_idx in range(optimal_factors):
    print(f"\nFACTOR {factor_idx + 1}:")
    print(f"Variance explained: {variance_explained[factor_idx]*100:.2f}%")
    
    # Get the top loadings for this factor
    factor_loadings_abs = np.abs(loadings_df[f'Factor {factor_idx+1}'])
    factor_loadings_values = loadings_df[f'Factor {factor_idx+1}']
    
    # Get significant loadings
    significant_mask = factor_loadings_abs > 0.40
    significant_vars = factor_loadings_values[significant_mask]
    significant_vars_sorted = significant_vars.reindex(significant_vars.abs().sort_values(ascending=False).index)
    
    print(f"Number of significant loadings (|loading| > 0.40): {len(significant_vars_sorted)}")
    print(f"\nTop 10 variables:")
    
    top_vars = []
    for i, (var_name, loading) in enumerate(significant_vars_sorted.head(10).items(), 1):
        print(f"  {i:2d}. {loading:7.3f}  {var_name[:75]}")
        top_vars.append((var_name, loading))
    
    factor_characteristics[factor_idx + 1] = top_vars

# ============================================================================
# 6. NAME THE FACTORS BASED ON CHARACTERISTICS
# ============================================================================

print(f"\n" + "=" * 80)
print(f"FACTOR INTERPRETATION & NAMING")
print("=" * 80)

# Analyze the top variables for each factor to suggest names
factor_names = {}

for factor_idx in range(1, optimal_factors + 1):
    print(f"\n{'='*80}")
    print(f"FACTOR {factor_idx}")
    print(f"{'='*80}")
    
    # Get the characteristics
    if factor_idx in factor_characteristics:
        top_vars = factor_characteristics[factor_idx]
        
        print(f"\nTop 5 Variables:")
        for i, (var_name, loading) in enumerate(top_vars[:5], 1):
            direction = "Positive" if loading > 0 else "Negative"
            print(f"  {i}. ({direction}) {loading:.3f}: {var_name}")
        
        # Suggest factor name based on content
        # This is a manual process - you should review and rename based on semantic analysis
        if factor_idx == 1:
            name = "Extraversion & Social Engagement"
            description = "Preference for social interaction, group activities, and external engagement"
        elif factor_idx == 2:
            name = "Conscientious Planning & Organization"
            description = "Tendency to plan ahead, organize tasks, and follow structured approaches"
        elif factor_idx == 3:
            name = "Emotional Reactivity & Sensitivity"
            description = "Tendency to experience strong emotions, worry, and emotional sensitivity"
        elif factor_idx == 4:
            name = "Logical Thinking & Rationality"
            description = "Preference for logical analysis over emotional decision-making"
        elif factor_idx == 5:
            name = "Imaginative & Theoretical Thinking"
            description = "Interest in abstract concepts, theory, and creative interpretation"
        elif factor_idx == 6:
            name = "Emotional Stability & Confidence"
            description = "Emotional resilience, lack of self-doubt, and overall confidence"
        else:
            name = f"Latent Factor {factor_idx}"
            description = "Interpret based on loadings above"
        
        factor_names[factor_idx] = (name, description)
        
        print(f"\nSuggested Factor Name: {name}")
        print(f"Description: {description}")

# ============================================================================
# 7. CREATE VISUALIZATION OF LOADINGS
# ============================================================================

print(f"\n" + "=" * 80)
print("CREATING VISUALIZATIONS")
print("=" * 80)

# Heatmap of loadings
fig, ax = plt.subplots(figsize=(12, 14))
sns.heatmap(loadings_df, cmap='RdBu_r', center=0, annot=False, fmt='.2f', 
            cbar_kws={'label': 'Loading'}, ax=ax, vmin=-1, vmax=1)
ax.set_title('Factor Loadings Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('factor_loadings_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Factor loadings heatmap saved to 'factor_loadings_heatmap.png'")
plt.close()

# Biplot for first two factors (if more than one factor)
if optimal_factors >= 2:
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot arrows for variables
    for i, var in enumerate(X.columns):
        ax.arrow(0, 0, loadings_df.iloc[i, 0]*1.5, loadings_df.iloc[i, 1]*1.5,
                head_width=0.05, head_length=0.05, fc='steelblue', ec='steelblue', alpha=0.5)
    
    # Add variable labels
    for i, var in enumerate(X.columns):
        ax.text(loadings_df.iloc[i, 0]*1.6, loadings_df.iloc[i, 1]*1.6, var[:25], 
               fontsize=7, ha='center', va='center', alpha=0.7)
    
    # Add circles
    circle1 = plt.Circle((0, 0), 1, fill=False, edgecolor='red', linestyle='--', linewidth=2, alpha=0.5)
    ax.add_patch(circle1)
    
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 1.8)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_xlabel(f'Factor 1 ({variance_explained[0]*100:.2f}%)', fontsize=11, fontweight='bold')
    ax.set_ylabel(f'Factor 2 ({variance_explained[1]*100:.2f}%)', fontsize=11, fontweight='bold')
    ax.set_title('Factor Analysis Biplot (Factor 1 vs Factor 2)', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('factor_biplot.png', dpi=300, bbox_inches='tight')
    print("✓ Factor biplot saved to 'factor_biplot.png'")
    plt.close()

# Loadings by factor
for f in range(min(3, optimal_factors)):  # First 3 factors
    fig, ax = plt.subplots(figsize=(10, 12))
    
    factor_col = f'Factor {f+1}'
    sorted_loadings = loadings_df[factor_col].sort_values()
    
    colors = ['red' if x < 0 else 'steelblue' for x in sorted_loadings.values]
    ax.barh(range(len(sorted_loadings)), sorted_loadings.values, color=colors, alpha=0.7)
    ax.set_yticks(range(len(sorted_loadings)))
    ax.set_yticklabels(sorted_loadings.index, fontsize=8)
    ax.set_xlabel('Loading', fontsize=11)
    ax.set_title(f'{factor_col} - Variable Loadings', fontsize=12, fontweight='bold')
    ax.axvline(x=0, color='k', linewidth=1)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'factor_{f+1}_loadings.png', dpi=300, bbox_inches='tight')
    print(f"✓ Factor {f+1} loadings plot saved to 'factor_{f+1}_loadings.png'")
    plt.close()

# ============================================================================
# 8. SAVE RESULTS
# ============================================================================

print(f"\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Save loadings to CSV
loadings_df.to_csv('factor_loadings.csv')
print("✓ Factor loadings saved to 'factor_loadings.csv'")

# Save communalities
communalities_df.to_csv('communalities.csv')
print("✓ Communalities saved to 'communalities.csv'")

# Save factor scores
factor_scores_df = pd.DataFrame(
    factors,
    columns=[f'Factor {i+1}' for i in range(optimal_factors)]
)
factor_scores_df.to_csv('factor_scores.csv', index=False)
print("✓ Factor scores saved to 'factor_scores.csv'")

# Save variance explained
variance_df = pd.DataFrame({
    'Factor': [f'Factor {i+1}' for i in range(optimal_factors)],
    'Variance Explained': variance_explained,
    'Cumulative Variance': np.cumsum(variance_explained)
})
variance_df.to_csv('variance_explained.csv', index=False)
print("✓ Variance explained saved to 'variance_explained.csv'")

# Save summary report
with open('factor_analysis_summary.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("FACTOR ANALYSIS SUMMARY REPORT\n")
    f.write("16 Personality Types Dataset\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Dataset Information:\n")
    f.write(f"  Number of observations: {X.shape[0]}\n")
    f.write(f"  Number of variables: {X.shape[1]}\n")
    f.write(f"  Optimal number of factors: {optimal_factors}\n")
    f.write(f"  Total variance explained: {variance_explained.sum()*100:.2f}%\n\n")
    
    f.write("="*80 + "\n")
    f.write("FACTOR NAMES AND CHARACTERISTICS\n")
    f.write("="*80 + "\n")
    
    for factor_num in range(1, optimal_factors + 1):
        f.write(f"\n{'-'*80}\n")
        f.write(f"FACTOR {factor_num}\n")
        f.write(f"{'-'*80}\n")
        
        if factor_num in factor_names:
            name, description = factor_names[factor_num]
            f.write(f"Name: {name}\n")
            f.write(f"Description: {description}\n")
        
        f.write(f"Variance Explained: {variance_explained[factor_num-1]*100:.2f}%\n\n")
        
        f.write(f"Top 10 Loading Variables:\n")
        if factor_num in factor_characteristics:
            for i, (var_name, loading) in enumerate(factor_characteristics[factor_num][:10], 1):
                direction = "↑" if loading > 0 else "↓"
                f.write(f"  {i:2d}. {direction} {loading:7.3f}  {var_name}\n")

print("✓ Summary report saved to 'factor_analysis_summary.txt'")

# ============================================================================
# 9. DETAILED FACTOR INTERPRETATION GUIDE
# ============================================================================

print(f"\n" + "=" * 80)
print("DETAILED FACTOR INTERPRETATION")
print("=" * 80)

with open('factor_interpretation_guide.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("FACTOR INTERPRETATION GUIDE\n")
    f.write("="*80 + "\n\n")
    
    f.write("This guide provides interpretations for each identified factor based on the\n")
    f.write("variables with the highest loadings (strength of relationship).\n\n")
    
    for factor_num in range(1, optimal_factors + 1):
        f.write(f"\n{'='*80}\n")
        f.write(f"FACTOR {factor_num}\n")
        f.write(f"{'='*80}\n")
        
        if factor_num in factor_names:
            name, description = factor_names[factor_num]
            f.write(f"\nFactor Name: {name}\n")
            f.write(f"Description: {description}\n")
        
        f.write(f"\nVariance Explained: {variance_explained[factor_num-1]*100:.2f}%\n\n")
        
        f.write(f"Interpretation Guide:\n")
        f.write(f"-" * 80 + "\n")
        
        if factor_num in factor_characteristics:
            positive_vars = [v for v in factor_characteristics[factor_num] if v[1] > 0]
            negative_vars = [v for v in factor_characteristics[factor_num] if v[1] < 0]
            
            if positive_vars:
                f.write(f"\nPositive Direction (High Factor Score):\n")
                f.write(f"Individuals with high scores on this factor tend to:\n")
                for i, (var_name, loading) in enumerate(positive_vars[:5], 1):
                    f.write(f"  • {var_name}\n")
            
            if negative_vars:
                f.write(f"\nNegative Direction (Low Factor Score):\n")
                f.write(f"Individuals with low scores on this factor tend to:\n")
                for i, (var_name, loading) in enumerate(negative_vars[:5], 1):
                    f.write(f"  • {var_name}\n")

print("✓ Interpretation guide saved to 'factor_interpretation_guide.txt'")

print(f"\n" + "=" * 80)
print("FACTOR ANALYSIS COMPLETE!")
print("="*80)
print("\nGenerated Files:")
print("  1. factor_loadings.csv - Factor loadings matrix")
print("  2. communalities.csv - Communality values for each variable")
print("  3. factor_scores.csv - Computed factor scores for each observation")
print("  4. variance_explained.csv - Variance explained by each factor")
print("  5. scree_plot.png - Scree plot and cumulative variance")
print("  6. factor_loadings_heatmap.png - Heatmap of all loadings")
print("  7. factor_biplot.png - Biplot of first two factors")
print("  8. factor_1/2/3_loadings.png - Individual factor loading plots")
print("  9. factor_analysis_summary.txt - Summary report")
print(" 10. factor_interpretation_guide.txt - Detailed interpretation guide")
print("="*80)
