"""
Cluster Analysis on Factor Scores from Cat Breeds Dataset
This script performs clustering on the factor scores to identify distinct cat breed groups.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. RELOAD DATA AND PERFORM FACTOR ANALYSIS
# ============================================================================

print("=" * 80)
print("CLUSTER ANALYSIS ON CAT BREEDS BASED ON FACTOR SCORES")
print("=" * 80)

# Load the CSV data
df = pd.read_csv('cat_breeds.csv')

# Select numeric features for analysis
numeric_cols = [
    'min_life_expectancy', 'max_life_expectancy', 'min_weight', 'max_weight',
    'family_friendly', 'shedding', 'general_health', 'playfulness',
    'children_friendly', 'grooming', 'intelligence', 'other_pets_friendly'
]

# Create a subset with numeric features
df_numeric = df[numeric_cols].copy()
df_numeric = df_numeric.dropna()

print(f"\nDataset: {len(df_numeric)} cat breeds")
print(f"Features: {len(numeric_cols)}")

# Standardize the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_numeric)
df_scaled = pd.DataFrame(df_scaled, columns=numeric_cols)

# Perform Factor Analysis with 4 factors
print("\nPerforming Factor Analysis with 4 factors...")
fa = FactorAnalysis(n_components=4, random_state=42, max_iter=1000)
fa.fit(df_scaled.values)

# Get factor scores
factor_scores = fa.transform(df_scaled.values)
factor_scores_df = pd.DataFrame(
    factor_scores,
    columns=['Factor_1_Size', 'Factor_2_Longevity', 'Factor_3_Sociability', 'Factor_4_Maintenance'],
    index=df['name'][:len(factor_scores)]
)

print("Factor scores calculated")
print(f"Factor scores shape: {factor_scores_df.shape}")

# ============================================================================
# 2. DETERMINE OPTIMAL NUMBER OF CLUSTERS
# ============================================================================

print("\n" + "=" * 80)
print("DETERMINING OPTIMAL NUMBER OF CLUSTERS")
print("=" * 80)

# Elbow Method
inertias = []
silhouette_scores = []
K_range = range(2, 11)

from sklearn.metrics import silhouette_score

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(factor_scores)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(factor_scores, kmeans.labels_))

print(f"\nElbow Method - Inertias:")
for k, inertia in zip(K_range, inertias):
    print(f"  K={k}: {inertia:.2f}")

print(f"\nSilhouette Scores:")
for k, score in zip(K_range, silhouette_scores):
    print(f"  K={k}: {score:.4f}")

# Find optimal K using silhouette score
optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"\n{'*' * 80}")
print(f"OPTIMAL NUMBER OF CLUSTERS: {optimal_k} (based on highest silhouette score)")
print(f"{'*' * 80}")

# ============================================================================
# 3. PERFORM K-MEANS CLUSTERING
# ============================================================================

print("\n" + "=" * 80)
print(f"PERFORMING K-MEANS CLUSTERING WITH {optimal_k} CLUSTERS")
print("=" * 80)

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(factor_scores)

# Add cluster labels to the dataframe
factor_scores_df['Cluster'] = cluster_labels

print(f"\nCluster Distribution:")
cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
for cluster_id, count in cluster_counts.items():
    percentage = (count / len(cluster_labels)) * 100
    print(f"  Cluster {cluster_id}: {count} breeds ({percentage:.1f}%)")

# ============================================================================
# 4. ANALYZE CLUSTER CHARACTERISTICS
# ============================================================================

print("\n" + "=" * 80)
print("CLUSTER CHARACTERISTICS ANALYSIS")
print("=" * 80)

cluster_profiles = []

for cluster_id in range(optimal_k):
    print(f"\n{'*' * 80}")
    print(f"CLUSTER {cluster_id}")
    print(f"{'*' * 80}")
    
    cluster_mask = factor_scores_df['Cluster'] == cluster_id
    cluster_data = factor_scores_df[cluster_mask]
    
    print(f"\nNumber of breeds: {len(cluster_data)}")
    print(f"Breeds in cluster:")
    
    breed_names = cluster_data.index.tolist()
    for i, breed in enumerate(breed_names, 1):
        print(f"  {i:2d}. {breed}")
    
    # Calculate mean factor scores for the cluster
    print(f"\nMean Factor Scores:")
    factor_means = cluster_data[['Factor_1_Size', 'Factor_2_Longevity', 
                                  'Factor_3_Sociability', 'Factor_4_Maintenance']].mean()
    
    profile_dict = {'Cluster': cluster_id}
    
    print(f"  Factor 1 (Physical Size): {factor_means['Factor_1_Size']:7.3f}", end="")
    if factor_means['Factor_1_Size'] > 0.3:
        print(" → LARGE BREEDS")
        profile_dict['Size'] = 'Large'
    elif factor_means['Factor_1_Size'] < -0.3:
        print(" → SMALL BREEDS")
        profile_dict['Size'] = 'Small'
    else:
        print(" → MEDIUM BREEDS")
        profile_dict['Size'] = 'Medium'
    
    print(f"  Factor 2 (Longevity): {factor_means['Factor_2_Longevity']:7.3f}", end="")
    if factor_means['Factor_2_Longevity'] > 0.3:
        print(" → LONG-LIVED")
        profile_dict['Longevity'] = 'Long-lived'
    elif factor_means['Factor_2_Longevity'] < -0.3:
        print(" → SHORT-LIVED")
        profile_dict['Longevity'] = 'Short-lived'
    else:
        print(" → MODERATE LIFESPAN")
        profile_dict['Longevity'] = 'Moderate'
    
    print(f"  Factor 3 (Sociability): {factor_means['Factor_3_Sociability']:7.3f}", end="")
    if factor_means['Factor_3_Sociability'] > 0.3:
        print(" → INDEPENDENT/ALOOF")
        profile_dict['Sociability'] = 'Independent'
    elif factor_means['Factor_3_Sociability'] < -0.3:
        print(" → HIGHLY SOCIAL/PLAYFUL")
        profile_dict['Sociability'] = 'Social'
    else:
        print(" → MODERATELY SOCIAL")
        profile_dict['Sociability'] = 'Moderate'
    
    print(f"  Factor 4 (Maintenance): {factor_means['Factor_4_Maintenance']:7.3f}", end="")
    if factor_means['Factor_4_Maintenance'] > 0.3:
        print(" → HIGH SHEDDING, LOW GROOMING")
        profile_dict['Maintenance'] = 'Low Maintenance'
    elif factor_means['Factor_4_Maintenance'] < -0.3:
        print(" → LOW SHEDDING, HIGH GROOMING")
        profile_dict['Maintenance'] = 'High Maintenance'
    else:
        print(" → BALANCED")
        profile_dict['Maintenance'] = 'Balanced'
    
    cluster_profiles.append(profile_dict)
    
    # Get actual feature means for context
    print(f"\nActual Feature Means:")
    # Create mapping between factor_scores_df index and df_numeric
    cluster_indices = []
    for breed_name in breed_names:
        # Find the position in df_numeric that corresponds to this breed
        match_indices = [i for i, name in enumerate(df['name'][:len(df_numeric)]) if name == breed_name]
        if match_indices:
            cluster_indices.append(match_indices[0])
    
    original_data = df_numeric.iloc[cluster_indices]
    
    print(f"  Weight range: {original_data['min_weight'].mean():.1f}-{original_data['max_weight'].mean():.1f} kg")
    print(f"  Life expectancy: {original_data['min_life_expectancy'].mean():.1f}-{original_data['max_life_expectancy'].mean():.1f} years")
    print(f"  Family friendly: {original_data['family_friendly'].mean():.1f}/5")
    print(f"  Playfulness: {original_data['playfulness'].mean():.1f}/5")
    print(f"  Children friendly: {original_data['children_friendly'].mean():.1f}/5")
    print(f"  Shedding: {original_data['shedding'].mean():.1f}/5")
    print(f"  Grooming: {original_data['grooming'].mean():.1f}/5")
    print(f"  Intelligence: {original_data['intelligence'].mean():.1f}/5")
    print(f"  Other pets friendly: {original_data['other_pets_friendly'].mean():.1f}/5")
    print(f"  General health: {original_data['general_health'].mean():.1f}/5")

# ============================================================================
# 5. NAME CLUSTERS AND CREATE DESCRIPTIONS
# ============================================================================

print("\n" + "=" * 80)
print("CLUSTER NAMING AND CHARACTERIZATION")
print("=" * 80)

cluster_names = {}
cluster_descriptions = {}

# Define cluster names based on profiles
for profile in cluster_profiles:
    cluster_id = profile['Cluster']
    cluster_data = factor_scores_df[factor_scores_df['Cluster'] == cluster_id]
    
    size = profile['Size']
    longevity = profile['Longevity']
    sociability = profile['Sociability']
    maintenance = profile['Maintenance']
    
    # Create descriptive names based on cluster characteristics
    if size == 'Large' and longevity == 'Long-lived':
        if sociability == 'Social':
            cluster_names[cluster_id] = "Majestic Companions"
        else:
            cluster_names[cluster_id] = "Robust & Reserved"
    elif size == 'Large' and longevity == 'Short-lived':
        cluster_names[cluster_id] = "Laid-back Giants"
    elif size == 'Small' and sociability == 'Social':
        cluster_names[cluster_id] = "Spirited Miniatures"
    elif size == 'Small' and sociability == 'Moderate':
        cluster_names[cluster_id] = "Playful Sprites"
    elif size == 'Small' and maintenance == 'Low Maintenance':
        cluster_names[cluster_id] = "Easy-care Sprites"
    elif size == 'Medium' and sociability == 'Independent':
        cluster_names[cluster_id] = "Sophisticated Independents"
    elif size == 'Medium' and sociability == 'Social':
        cluster_names[cluster_id] = "Balanced Charmers"
    elif sociability == 'Independent' and maintenance == 'High Maintenance':
        cluster_names[cluster_id] = "Elegant Aloofers"
    elif sociability == 'Social' and maintenance == 'High Maintenance':
        cluster_names[cluster_id] = "High-Maintenance Charmers"
    elif maintenance == 'Low Maintenance':
        cluster_names[cluster_id] = "Low-Maintenance Cuties"
    else:
        cluster_names[cluster_id] = f"Cluster {cluster_id}: {size} {sociability}"

print("\nCluster Names and Descriptions:\n")

for cluster_id in range(optimal_k):
    cluster_data = factor_scores_df[factor_scores_df['Cluster'] == cluster_id]
    breed_names = cluster_data.index.tolist()
    
    cluster_name = cluster_names[cluster_id]
    
    # Get profile details
    profile = [p for p in cluster_profiles if p['Cluster'] == cluster_id][0]
    size = profile['Size']
    longevity = profile['Longevity']
    sociability = profile['Sociability']
    maintenance = profile['Maintenance']
    
    # Create detailed descriptions
    print(f"{'=' * 80}")
    print(f"CLUSTER {cluster_id}: {cluster_name.upper()}")
    print(f"{'=' * 80}")
    print(f"\nNumber of breeds: {len(breed_names)}")
    print(f"\nCluster Composition:")
    print(f"  • Size: {size}")
    print(f"  • Lifespan: {longevity}")
    print(f"  • Temperament: {sociability}")
    print(f"  • Maintenance: {maintenance}")
    
    # Get actual stats
    cluster_indices = []
    for breed_name in breed_names:
        match_indices = [i for i, name in enumerate(df['name'][:len(df_numeric)]) if name == breed_name]
        if match_indices:
            cluster_indices.append(match_indices[0])
    
    original_data = df_numeric.iloc[cluster_indices]
    
    strengths = []
    weaknesses = []
    
    # Determine strengths and weaknesses
    avg_playfulness = original_data['playfulness'].mean()
    avg_family = original_data['family_friendly'].mean()
    avg_children = original_data['children_friendly'].mean()
    avg_intelligence = original_data['intelligence'].mean()
    avg_health = original_data['general_health'].mean()
    avg_grooming = original_data['grooming'].mean()
    avg_shedding = original_data['shedding'].mean()
    avg_other_pets = original_data['other_pets_friendly'].mean()
    
    if avg_playfulness >= 4.0:
        strengths.append("excellent playfulness and energy levels")
    elif avg_playfulness >= 3.5:
        strengths.append("good playfulness")
    
    if avg_family >= 4.0 and avg_children >= 4.0:
        strengths.append("highly family and child-friendly")
    elif avg_family >= 4.0:
        strengths.append("family-friendly nature")
    
    if avg_intelligence >= 4.0:
        strengths.append("strong intelligence and trainability")
    
    if avg_health >= 4.0:
        strengths.append("excellent general health")
    
    if avg_other_pets >= 4.0:
        strengths.append("great compatibility with other pets")
    
    if avg_grooming >= 4.5:
        weaknesses.append("high grooming requirements")
    
    if avg_shedding >= 4.0:
        weaknesses.append("significant shedding")
    
    if avg_playfulness <= 2.5:
        weaknesses.append("limited playfulness and energy")
    
    if avg_family <= 3.0:
        weaknesses.append("lower family-friendliness")
    
    if avg_children <= 3.0:
        weaknesses.append("not ideal for homes with children")
    
    if avg_intelligence <= 2.5:
        weaknesses.append("lower intelligence levels")
    
    if avg_other_pets <= 2.5:
        weaknesses.append("poor compatibility with other pets")
    
    if not strengths:
        strengths.append("balanced characteristics across all attributes")
    
    if not weaknesses:
        # Instead of "no significant weaknesses", provide contextual alternatives
        if avg_health <= 2.5:
            weaknesses.append("potential health concerns")
        elif avg_grooming <= 2.0 and avg_shedding >= 4.0:
            weaknesses.append("moderate grooming despite high shedding")
        elif avg_playfulness <= 2.0:
            weaknesses.append("lower activity and engagement levels")
        elif avg_intelligence <= 2.5:
            weaknesses.append("limited trainability")
        else:
            weaknesses.append("relatively consistent care requirements with few drawbacks")
    
    # Create three-sentence descriptions
    overview_sentence = f"The {cluster_name} cluster comprises {len(breed_names)} {size.lower()} cat breeds with {longevity.lower()} lifespans and a {sociability.lower()} temperament."
    
    strength_sentence = f"These breeds excel in {', and '.join(strengths[:2])}."
    
    weakness_sentence = f"However, they tend to struggle with {', and '.join(weaknesses[:2])}."
    
    cluster_descriptions[cluster_id] = {
        'name': cluster_name,
        'overview': overview_sentence,
        'strengths': strength_sentence,
        'weaknesses': weakness_sentence,
        'breeds': breed_names
    }
    
    print(f"\nDescription:")
    print(f"  1. {overview_sentence}")
    print(f"  2. {strength_sentence}")
    print(f"  3. {weakness_sentence}")
    
    print(f"\nBreeds in this cluster ({len(breed_names)}):")
    for i, breed in enumerate(breed_names, 1):
        print(f"  {i:2d}. {breed}")

# ============================================================================
# 6. HIERARCHICAL CLUSTERING
# ============================================================================

print("\n" + "=" * 80)
print("HIERARCHICAL CLUSTERING ANALYSIS")
print("=" * 80)

# Perform hierarchical clustering
linkage_matrix = linkage(factor_scores, method='ward')
print("\nHierarchical clustering linkage matrix computed (Ward method)")

# ============================================================================
# 7. CREATE VISUALIZATIONS
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Plot 1: Elbow Method
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('Number of Clusters (K)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Inertia', fontsize=11, fontweight='bold')
ax1.set_title('Elbow Method', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal K={optimal_k}')
ax1.legend()

# Plot 2: Silhouette Scores
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
ax2.set_xlabel('Number of Clusters (K)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Silhouette Score', fontsize=11, fontweight='bold')
ax2.set_title('Silhouette Score Analysis', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal K={optimal_k}')
ax2.legend()

# Plot 3: Dendrogram
ax3 = fig.add_subplot(gs[1, :])
dendrogram(linkage_matrix, labels=factor_scores_df.index, ax=ax3, leaf_font_size=8)
ax3.set_title('Hierarchical Clustering Dendrogram', fontsize=12, fontweight='bold')
ax3.set_xlabel('Breed', fontsize=11, fontweight='bold')
ax3.set_ylabel('Distance', fontsize=11, fontweight='bold')
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=90, fontsize=7)

# Plot 4: 2D Scatter - Factor 1 vs Factor 2
ax4 = fig.add_subplot(gs[2, 0])
scatter1 = ax4.scatter(factor_scores[:, 0], factor_scores[:, 1], 
                       c=cluster_labels, cmap='viridis', s=100, alpha=0.6, edgecolors='black')
ax4.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
           c='red', marker='X', s=300, edgecolors='black', linewidth=2, label='Centroids')
ax4.set_xlabel('Factor 1: Physical Size', fontsize=11, fontweight='bold')
ax4.set_ylabel('Factor 2: Longevity', fontsize=11, fontweight='bold')
ax4.set_title('Clusters: Size vs Longevity', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend()
plt.colorbar(scatter1, ax=ax4, label='Cluster')

# Plot 5: 2D Scatter - Factor 3 vs Factor 4
ax5 = fig.add_subplot(gs[2, 1])
scatter2 = ax5.scatter(factor_scores[:, 2], factor_scores[:, 3],
                       c=cluster_labels, cmap='viridis', s=100, alpha=0.6, edgecolors='black')
ax5.scatter(kmeans.cluster_centers_[:, 2], kmeans.cluster_centers_[:, 3],
           c='red', marker='X', s=300, edgecolors='black', linewidth=2, label='Centroids')
ax5.set_xlabel('Factor 3: Sociability', fontsize=11, fontweight='bold')
ax5.set_ylabel('Factor 4: Maintenance', fontsize=11, fontweight='bold')
ax5.set_title('Clusters: Sociability vs Maintenance', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.legend()
plt.colorbar(scatter2, ax=ax5, label='Cluster')

plt.savefig('/mnt/user-data/outputs/cluster_analysis_results.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved as 'cluster_analysis_results.png'")
plt.show()

# ============================================================================
# 8. CREATE CLUSTER PROFILES HEATMAP
# ============================================================================

print("\nCreating cluster profiles heatmap...")

# Calculate mean factor scores for each cluster
cluster_mean_factors = factor_scores_df.groupby('Cluster')[
    ['Factor_1_Size', 'Factor_2_Longevity', 'Factor_3_Sociability', 'Factor_4_Maintenance']
].mean()

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(cluster_mean_factors.T, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
            cbar_kws={'label': 'Mean Factor Score'}, ax=ax, vmin=-1, vmax=1)
ax.set_title('Cluster Profiles: Mean Factor Scores', fontsize=13, fontweight='bold')
ax.set_xlabel('Cluster', fontsize=11, fontweight='bold')
ax.set_ylabel('Factor', fontsize=11, fontweight='bold')
ax.set_xticklabels([f"C{i}\n{cluster_names[i]}" for i in range(optimal_k)], fontsize=10)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/cluster_profiles_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Cluster profiles heatmap saved as 'cluster_profiles_heatmap.png'")
plt.show()

# ============================================================================
# 9. EXPORT RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("EXPORTING RESULTS")
print("=" * 80)

# Save cluster assignments with breed names
cluster_assignment_df = pd.DataFrame({
    'Breed': factor_scores_df.index,
    'Cluster': factor_scores_df['Cluster'],
    'Cluster_Name': factor_scores_df['Cluster'].map(cluster_names),
    'Factor_1_Size': factor_scores[:, 0],
    'Factor_2_Longevity': factor_scores[:, 1],
    'Factor_3_Sociability': factor_scores[:, 2],
    'Factor_4_Maintenance': factor_scores[:, 3]
})

cluster_assignment_df.to_csv('/mnt/user-data/outputs/breed_cluster_assignments.csv', index=False)
print("✓ Cluster assignments saved as 'breed_cluster_assignments.csv'")

# Save cluster summary
cluster_summary = []
for cluster_id in range(optimal_k):
    cluster_data = factor_scores_df[factor_scores_df['Cluster'] == cluster_id]
    breed_names = cluster_data.index.tolist()
    
    cluster_summary.append({
        'Cluster_ID': cluster_id,
        'Cluster_Name': cluster_names[cluster_id],
        'Number_of_Breeds': len(breed_names),
        'Overview': cluster_descriptions[cluster_id]['overview'],
        'Strengths': cluster_descriptions[cluster_id]['strengths'],
        'Weaknesses': cluster_descriptions[cluster_id]['weaknesses'],
        'Breeds': ', '.join(breed_names)
    })

cluster_summary_df = pd.DataFrame(cluster_summary)
cluster_summary_df.to_csv('/mnt/user-data/outputs/cluster_summary.csv', index=False)
print("✓ Cluster summary saved as 'cluster_summary.csv'")

# ============================================================================
# 10. GENERATE COMPREHENSIVE REPORT
# ============================================================================

report_text = f"""
CLUSTER ANALYSIS REPORT - CAT BREEDS DATASET
{'=' * 80}

ANALYSIS OVERVIEW
{'=' * 80}
Based on the 4 extracted factors from the previous factor analysis, a K-means 
clustering analysis was performed to identify distinct groups of cat breeds.

Number of clusters: {optimal_k}
Clustering method: K-means (optimized using Silhouette score)
Silhouette score: {max(silhouette_scores):.4f}

{'=' * 80}
CLUSTER DEFINITIONS AND DESCRIPTIONS
{'=' * 80}

"""

for cluster_id in range(optimal_k):
    cluster_data = factor_scores_df[factor_scores_df['Cluster'] == cluster_id]
    breed_names = sorted(cluster_data.index.tolist())
    
    report_text += f"\n{'=' * 80}\n"
    report_text += f"CLUSTER {cluster_id}: {cluster_names[cluster_id].upper()}\n"
    report_text += f"{'=' * 80}\n\n"
    
    report_text += f"Cluster Size: {len(breed_names)} breeds\n\n"
    
    report_text += "Cluster Description:\n"
    report_text += f"  1. {cluster_descriptions[cluster_id]['overview']}\n"
    report_text += f"  2. {cluster_descriptions[cluster_id]['strengths']}\n"
    report_text += f"  3. {cluster_descriptions[cluster_id]['weaknesses']}\n\n"
    
    report_text += f"Mean Factor Scores:\n"
    cluster_factors = factor_scores_df[factor_scores_df['Cluster'] == cluster_id][
        ['Factor_1_Size', 'Factor_2_Longevity', 'Factor_3_Sociability', 'Factor_4_Maintenance']
    ].mean()
    
    report_text += f"  • Factor 1 (Physical Size): {cluster_factors['Factor_1_Size']:7.3f}\n"
    report_text += f"  • Factor 2 (Longevity): {cluster_factors['Factor_2_Longevity']:7.3f}\n"
    report_text += f"  • Factor 3 (Sociability): {cluster_factors['Factor_3_Sociability']:7.3f}\n"
    report_text += f"  • Factor 4 (Maintenance): {cluster_factors['Factor_4_Maintenance']:7.3f}\n\n"
    
    report_text += f"Breeds in this cluster:\n"
    for i, breed in enumerate(breed_names, 1):
        report_text += f"  {i:2d}. {breed}\n"
    report_text += "\n"

report_text += f"\n{'=' * 80}\n"
report_text += "CLUSTERING QUALITY METRICS\n"
report_text += f"{'=' * 80}\n\n"

report_text += f"Silhouette Scores for different K values:\n"
for k, score in zip(K_range, silhouette_scores):
    marker = " ← SELECTED" if k == optimal_k else ""
    report_text += f"  K={k}: {score:.4f}{marker}\n"

report_text += f"\nOptimal K: {optimal_k} (highest Silhouette score)\n"

with open('/mnt/user-data/outputs/cluster_analysis_report.txt', 'w') as f:
    f.write(report_text)

print("✓ Comprehensive report saved as 'cluster_analysis_report.txt'")

# ============================================================================
# 11. SUMMARY STATISTICS
# ============================================================================

print("\n" + "=" * 80)
print("CLUSTER ANALYSIS COMPLETE")
print("=" * 80)

print(f"\nSummary Statistics:")
print(f"  • Optimal number of clusters: {optimal_k}")
print(f"  • Best silhouette score: {max(silhouette_scores):.4f}")
print(f"  • Total breeds analyzed: {len(factor_scores_df)}")
print(f"\nCluster Distribution:")
for cluster_id in range(optimal_k):
    count = (cluster_labels == cluster_id).sum()
    percentage = (count / len(cluster_labels)) * 100
    print(f"  • {cluster_names[cluster_id]:40s}: {count:2d} breeds ({percentage:5.1f}%)")

print(f"\nFiles generated:")
print(f"  ✓ cluster_analysis.py (this script)")
print(f"  ✓ cluster_analysis_results.png")
print(f"  ✓ cluster_profiles_heatmap.png")
print(f"  ✓ breed_cluster_assignments.csv")
print(f"  ✓ cluster_summary.csv")
print(f"  ✓ cluster_analysis_report.txt")

print("\n" + "=" * 80)
