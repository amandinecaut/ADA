import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_cluster_name(strengths, weaknesses, factor_map):
    """Generate cluster name based on factor profile"""
    if not strengths and not weaknesses:
        return "Mixed Profile Cluster"
    
    if strengths:
        top_strength_factor = strengths[0][0]
        top_strength_name = factor_map.get(top_strength_factor, top_strength_factor)
        
        if 'Other Tournament' in top_strength_name:
            return "Tournament Specialists"
        elif 'Club Match' in top_strength_name and 'Playing Time' in top_strength_name:
            return "Club Workhorses"
        elif 'National Team' in top_strength_name and 'Selection' in top_strength_name:
            return "National Team Leaders"
        elif 'Attacking' in top_strength_name or 'Scoring' in top_strength_name:
            return "Attacking Specialists"
        elif 'Physical' in top_strength_name:
            return "Physically Dominant"
    
    return "Balanced Performers"

def generate_overview_sentence(cluster_id, profile, factor_map):
    """Generate overview sentence for cluster"""
    size = profile['size']
    size_desc = "large" if size > 25 else "medium" if size > 15 else "small"
    
    strengths = profile['strengths']
    if strengths:
        top_factor = factor_map.get(strengths[0][0], strengths[0][0])
        return f"This {size_desc} cluster of {size} players is characterized by excellence in {top_factor.lower()}."
    else:
        return f"This {size_desc} cluster of {size} players shows balanced but moderate performance across all factors."

def generate_strengths_sentence(strengths, factor_map):
    """Generate strengths sentence"""
    if not strengths:
        return "The cluster shows no particularly distinctive strengths."
    
    if len(strengths) >= 2:
        factor1 = factor_map.get(strengths[0][0], strengths[0][0]).lower()
        factor2 = factor_map.get(strengths[1][0], strengths[1][0]).lower()
        return f"Key strengths include {factor1} and {factor2}."
    else:
        factor = factor_map.get(strengths[0][0], strengths[0][0]).lower()
        return f"The primary strength of this cluster is {factor}."

def generate_weaknesses_sentence(weaknesses, factor_map):
    """Generate weaknesses sentence"""
    if not weaknesses:
        return "The cluster demonstrates no significant weaknesses."
    
    if len(weaknesses) >= 2:
        factor1 = factor_map.get(weaknesses[0][0], weaknesses[0][0]).lower()
        factor2 = factor_map.get(weaknesses[1][0], weaknesses[1][0]).lower()
        return f"Notable limitations include {factor1} and {factor2}."
    else:
        factor = factor_map.get(weaknesses[0][0], weaknesses[0][0]).lower()
        return f"A notable limitation is {factor}."

# ============================================================================
# 1. LOAD FACTOR SCORES
# ============================================================================
print("="*80)
print("CLUSTER ANALYSIS ON EXTRACTED FACTORS")
print("="*80)

factor_scores_df = pd.read_csv('/mnt/user-data/outputs/factor_scores.csv')

print(f"\nLoaded factor scores for {len(factor_scores_df)} players")

players = factor_scores_df['Player'].values
factor_columns = [col for col in factor_scores_df.columns if col.startswith('Factor')]
factor_data = factor_scores_df[factor_columns].values

# Standardize for clustering
scaler = StandardScaler()
factor_data_scaled = scaler.fit_transform(factor_data)

print(f"Factor data shape: {factor_data.shape}")

# ============================================================================
# 2. DETERMINE OPTIMAL NUMBER OF CLUSTERS
# ============================================================================
print("\n" + "="*80)
print("DETERMINING OPTIMAL NUMBER OF CLUSTERS")
print("="*80)

silhouette_scores = []
davies_bouldin_scores = []
calinski_harabasz_scores = []
inertias = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(factor_data_scaled)
    
    silhouette = silhouette_score(factor_data_scaled, cluster_labels)
    davies_bouldin = davies_bouldin_score(factor_data_scaled, cluster_labels)
    calinski_harabasz = calinski_harabasz_score(factor_data_scaled, cluster_labels)
    inertia = kmeans.inertia_
    
    silhouette_scores.append(silhouette)
    davies_bouldin_scores.append(davies_bouldin)
    calinski_harabasz_scores.append(calinski_harabasz)
    inertias.append(inertia)
    
    print(f"k={k}: Silhouette={silhouette:.4f}, Davies-Bouldin={davies_bouldin:.4f}, Calinski-Harabasz={calinski_harabasz:.2f}")

optimal_k = list(K_range)[np.argmax(silhouette_scores)]
print(f"\n✓ Optimal number of clusters: {optimal_k}")

# ============================================================================
# 3. PERFORM K-MEANS CLUSTERING
# ============================================================================
print("\n" + "="*80)
print(f"K-MEANS CLUSTERING WITH {optimal_k} CLUSTERS")
print("="*80)

kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=20)
cluster_labels = kmeans_final.fit_predict(factor_data_scaled)

factor_scores_df['Cluster'] = cluster_labels

print(f"\nCluster distribution:")
for i in range(optimal_k):
    count = np.sum(cluster_labels == i)
    percentage = (count / len(cluster_labels)) * 100
    print(f"  Cluster {i}: {count} players ({percentage:.1f}%)")

# ============================================================================
# 4. ANALYZE CLUSTER CHARACTERISTICS
# ============================================================================
print("\n" + "="*80)
print("CLUSTER CHARACTERISTICS ANALYSIS")
print("="*80)

cluster_profiles = {}
factor_interpretations = {
    'Factor 1': 'Other Tournament Performance',
    'Factor 2': 'Club Match Frequency & Playing Time',
    'Factor 3': 'National Team Performance & Selection',
    'Factor 4': 'National Team Performance (Alternative)',
    'Factor 5': 'Club Match Results',
    'Factor 6': 'Club Playing Time & Attacking Contribution',
    'Factor 7': 'National Team Performance (Losses)',
    'Factor 8': 'Physical Attributes',
    'Factor 9': 'Scoring Performance'
}

for cluster_id in range(optimal_k):
    print(f"\n{'='*80}")
    print(f"CLUSTER {cluster_id}")
    print(f"{'='*80}")
    
    cluster_mask = cluster_labels == cluster_id
    cluster_members = players[cluster_mask]
    cluster_factor_scores = factor_data[cluster_mask]
    
    print(f"Cluster Size: {len(cluster_members)} players")
    print(f"Members (first 5): {', '.join(cluster_members[:5])}")
    
    mean_scores = np.mean(cluster_factor_scores, axis=1)
    std_scores = np.std(cluster_factor_scores, axis=1)
    
    factor_strengths = []
    factor_weaknesses = []
    
    print(f"\nFactor Profile (mean scores):")
    print(f"{'Factor':<15} {'Mean Score':<15} {'Interpretation':<30}")
    print("-" * 60)
    
    for i, factor in enumerate(factor_columns):
        overall_mean = np.mean(factor_data[:, i])
        cluster_mean = np.mean(cluster_factor_scores[:, i])
        difference = cluster_mean - overall_mean
        
        if difference > 0.5:
            interpretation = "Strong (High)"
            factor_strengths.append((factor, difference))
        elif difference < -0.5:
            interpretation = "Weak (Low)"
            factor_weaknesses.append((factor, difference))
        else:
            interpretation = "Average"
        
        print(f"{factor:<15} {difference:<15.4f} {interpretation:<30}")
    
    # Sort by magnitude
    factor_strengths.sort(key=lambda x: x[1], reverse=True)
    factor_weaknesses.sort(key=lambda x: x[1])
    
    cluster_profiles[cluster_id] = {
        'members': cluster_members,
        'size': len(cluster_members),
        'strengths': factor_strengths,
        'weaknesses': factor_weaknesses,
        'factor_data': cluster_factor_scores,
        'mean_scores': np.mean(cluster_factor_scores, axis=0)
    }

# ============================================================================
# 5. GENERATE CLUSTER NAMES AND DESCRIPTIONS
# ============================================================================
print("\n" + "="*80)
print("CLUSTER NAMING AND DESCRIPTIONS")
print("="*80)

cluster_names = {}
cluster_descriptions = {}

for cluster_id in range(optimal_k):
    profile = cluster_profiles[cluster_id]
    strengths = profile['strengths']
    weaknesses = profile['weaknesses']
    
    name = generate_cluster_name(strengths, weaknesses, factor_interpretations)
    cluster_names[cluster_id] = name
    
    overview = generate_overview_sentence(cluster_id, profile, factor_interpretations)
    strengths_sentence = generate_strengths_sentence(strengths, factor_interpretations)
    weaknesses_sentence = generate_weaknesses_sentence(weaknesses, factor_interpretations)
    
    description = f"{overview} {strengths_sentence} {weaknesses_sentence}"
    cluster_descriptions[cluster_id] = description
    
    print(f"\nCluster {cluster_id}: {name}")
    print(f"Description: {description}")

# ============================================================================
# 6. CREATE DETAILED CLUSTER PROFILES
# ============================================================================
print("\n" + "="*80)
print("DETAILED CLUSTER PROFILES")
print("="*80)

for cluster_id in range(optimal_k):
    profile = cluster_profiles[cluster_id]
    
    print(f"\n{'='*80}")
    print(f"CLUSTER {cluster_id}: {cluster_names[cluster_id].upper()}")
    print(f"{'='*80}")
    
    print(f"\n{cluster_descriptions[cluster_id]}")
    
    print(f"\nCluster Composition:")
    print(f"  • Size: {profile['size']} players ({(profile['size']/len(players))*100:.1f}%)")
    print(f"  • Members: {', '.join(list(profile['members'][:8]))}...")
    
    print(f"\nFactor Strengths (> 0.5 std above mean):")
    if profile['strengths']:
        for factor, score in profile['strengths'][:3]:
            print(f"  • {factor_interpretations.get(factor, factor)}: {score:.3f}")
    else:
        print("  • None identified")
    
    print(f"\nFactor Weaknesses (> 0.5 std below mean):")
    if profile['weaknesses']:
        for factor, score in profile['weaknesses'][:3]:
            print(f"  • {factor_interpretations.get(factor, factor)}: {score:.3f}")
    else:
        print("  • None identified")

# ============================================================================
# 7. CREATE VISUALIZATIONS
# ============================================================================
print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

fig = plt.figure(figsize=(18, 14))
colors = plt.cm.Set3(np.linspace(0, 1, optimal_k))

# 1. Elbow curve
ax1 = plt.subplot(3, 3, 1)
ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
ax1.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
ax1.set_xlabel('Number of Clusters', fontsize=11, fontweight='bold')
ax1.set_ylabel('Inertia', fontsize=11, fontweight='bold')
ax1.set_title('Elbow Curve', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Silhouette scores
ax2 = plt.subplot(3, 3, 2)
ax2.plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
ax2.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
ax2.set_xlabel('Number of Clusters', fontsize=11, fontweight='bold')
ax2.set_ylabel('Silhouette Score', fontsize=11, fontweight='bold')
ax2.set_title('Silhouette Score Analysis', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Davies-Bouldin scores
ax3 = plt.subplot(3, 3, 3)
ax3.plot(K_range, davies_bouldin_scores, 'mo-', linewidth=2, markersize=8)
ax3.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
ax3.set_xlabel('Number of Clusters', fontsize=11, fontweight='bold')
ax3.set_ylabel('Davies-Bouldin Index', fontsize=11, fontweight='bold')
ax3.set_title('Davies-Bouldin Index (Lower=Better)', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Cluster size distribution
ax4 = plt.subplot(3, 3, 4)
cluster_sizes = [np.sum(cluster_labels == i) for i in range(optimal_k)]
bars = ax4.bar(range(optimal_k), cluster_sizes, color=colors, edgecolor='black', alpha=0.7)
ax4.set_xlabel('Cluster', fontsize=11, fontweight='bold')
ax4.set_ylabel('Number of Players', fontsize=11, fontweight='bold')
ax4.set_title('Cluster Size Distribution', fontsize=12, fontweight='bold')
ax4.set_xticks(range(optimal_k))
ax4.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(cluster_sizes):
    ax4.text(i, v + 0.5, str(v), ha='center', fontweight='bold')

# 5. Factor 1 vs Factor 2
ax5 = plt.subplot(3, 3, 5)
for i in range(optimal_k):
    mask = cluster_labels == i
    ax5.scatter(factor_data_scaled[mask, 0], factor_data_scaled[mask, 1], 
               label=f'Cluster {i}', s=100, alpha=0.6, color=colors[i], edgecolor='black')
ax5.scatter(kmeans_final.cluster_centers_[:, 0], kmeans_final.cluster_centers_[:, 1], 
           marker='X', s=400, c='red', edgecolor='black', linewidth=2, label='Centroids')
ax5.set_xlabel('Factor 1 Score', fontsize=11, fontweight='bold')
ax5.set_ylabel('Factor 2 Score', fontsize=11, fontweight='bold')
ax5.set_title('Cluster Distribution (F1 vs F2)', fontsize=12, fontweight='bold')
ax5.legend(loc='best', fontsize=9)
ax5.grid(True, alpha=0.3)

# 6. Factor 3 vs Factor 4
ax6 = plt.subplot(3, 3, 6)
for i in range(optimal_k):
    mask = cluster_labels == i
    ax6.scatter(factor_data_scaled[mask, 2], factor_data_scaled[mask, 3], 
               label=f'Cluster {i}', s=100, alpha=0.6, color=colors[i], edgecolor='black')
ax6.scatter(kmeans_final.cluster_centers_[:, 2], kmeans_final.cluster_centers_[:, 3], 
           marker='X', s=400, c='red', edgecolor='black', linewidth=2)
ax6.set_xlabel('Factor 3 Score', fontsize=11, fontweight='bold')
ax6.set_ylabel('Factor 4 Score', fontsize=11, fontweight='bold')
ax6.set_title('Cluster Distribution (F3 vs F4)', fontsize=12, fontweight='bold')
ax6.legend(loc='best', fontsize=9)
ax6.grid(True, alpha=0.3)

# 7. Heatmap of cluster profiles
ax7 = plt.subplot(3, 3, 7)
cluster_means = np.array([cluster_profiles[i]['mean_scores'] for i in range(optimal_k)])
im = ax7.imshow(cluster_means, cmap='RdBu_r', aspect='auto', vmin=-2, vmax=2)
ax7.set_xticks(range(len(factor_columns)))
ax7.set_yticks(range(optimal_k))
ax7.set_xticklabels([f'F{i+1}' for i in range(len(factor_columns))], fontsize=9)
ax7.set_yticklabels([f'C{i}' for i in range(optimal_k)])
ax7.set_xlabel('Factors', fontsize=11, fontweight='bold')
ax7.set_ylabel('Cluster', fontsize=11, fontweight='bold')
ax7.set_title('Cluster Profile Heatmap', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax7, label='Mean Score')

for i in range(optimal_k):
    for j in range(len(factor_columns)):
        text = ax7.text(j, i, f'{cluster_means[i, j]:.1f}',
                       ha="center", va="center", color="black", fontsize=8, fontweight='bold')

# 8. Box plot of Factor 1
ax8 = plt.subplot(3, 3, 8)
box_data = [factor_data_scaled[cluster_labels == i, 0] for i in range(optimal_k)]
bp = ax8.boxplot(box_data, labels=[f'C{i}' for i in range(optimal_k)], patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax8.set_ylabel('Score', fontsize=11, fontweight='bold')
ax8.set_xlabel('Cluster', fontsize=11, fontweight='bold')
ax8.set_title('Factor 1 Distribution', fontsize=12, fontweight='bold')
ax8.grid(True, alpha=0.3, axis='y')

# 9. Cluster names summary
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')
summary_text = "CLUSTER SUMMARY\n" + "="*45 + "\n\n"
for i in range(optimal_k):
    summary_text += f"Cluster {i}: {cluster_names[i]}\n"
    summary_text += f"  Size: {cluster_sizes[i]} players\n\n"
ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=9,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/cluster_analysis_results.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: cluster_analysis_results.png")
plt.close()

# ============================================================================
# 8. SAVE RESULTS
# ============================================================================
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save cluster assignments
cluster_assignment_df = factor_scores_df[['Player', 'Cluster']].copy()
cluster_assignment_df['Cluster Name'] = cluster_assignment_df['Cluster'].map(cluster_names)
cluster_assignment_df.to_csv('/mnt/user-data/outputs/cluster_assignments.csv', index=False)
print("✓ Saved: cluster_assignments.csv")

# Save full results
full_results = factor_scores_df.copy()
full_results['Cluster Name'] = full_results['Cluster'].map(cluster_names)
full_results.to_csv('/mnt/user-data/outputs/player_factor_cluster_results.csv', index=False)
print("✓ Saved: player_factor_cluster_results.csv")

# Save cluster descriptions
with open('/mnt/user-data/outputs/cluster_descriptions.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("CLUSTER ANALYSIS RESULTS\n")
    f.write("="*80 + "\n\n")
    
    for i in range(optimal_k):
        f.write(f"CLUSTER {i}: {cluster_names[i]}\n")
        f.write(f"{'-'*80}\n\n")
        f.write(f"Description:\n{cluster_descriptions[i]}\n\n")
        
        profile = cluster_profiles[i]
        f.write(f"Cluster Size: {profile['size']} players ({(profile['size']/len(players))*100:.1f}%)\n\n")
        
        f.write(f"Members:\n")
        for j, player in enumerate(profile['members']):
            f.write(f"  {j+1}. {player}\n")
        f.write("\n")
        
        f.write(f"Strengths:\n")
        if profile['strengths']:
            for factor, score in profile['strengths']:
                f.write(f"  • {factor_interpretations.get(factor, factor)}: {score:.3f}\n")
        else:
            f.write("  • None identified\n")
        
        f.write(f"\nWeaknesses:\n")
        if profile['weaknesses']:
            for factor, score in profile['weaknesses']:
                f.write(f"  • {factor_interpretations.get(factor, factor)}: {score:.3f}\n")
        else:
            f.write("  • None identified\n")
        
        f.write("\n" + "="*80 + "\n\n")

print("✓ Saved: cluster_descriptions.txt")

# Save statistics
with open('/mnt/user-data/outputs/cluster_statistics.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("CLUSTER ANALYSIS STATISTICS\n")
    f.write("="*80 + "\n\n")
    
    f.write("OPTIMAL CLUSTER DETERMINATION\n")
    f.write("-"*80 + "\n")
    f.write(f"Silhouette Score: {silhouette_scores[optimal_k-2]:.4f}\n")
    f.write(f"Davies-Bouldin Index: {davies_bouldin_scores[optimal_k-2]:.4f}\n")
    f.write(f"Calinski-Harabasz Index: {calinski_harabasz_scores[optimal_k-2]:.2f}\n\n")
    
    f.write("QUALITY METRICS FOR ALL K VALUES\n")
    f.write("-"*80 + "\n")
    f.write(f"{'K':<5} {'Silhouette':<20} {'Davies-Bouldin':<20} {'Calinski-Harabasz':<20}\n")
    for i, k in enumerate(K_range):
        f.write(f"{k:<5} {silhouette_scores[i]:<20.4f} {davies_bouldin_scores[i]:<20.4f} {calinski_harabasz_scores[i]:<20.2f}\n")

print("✓ Saved: cluster_statistics.txt")

# ============================================================================
# 9. SUMMARY
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nCluster Summary:")
for i in range(optimal_k):
    print(f"  Cluster {i}: {cluster_names[i]} ({cluster_sizes[i]} players)")
print(f"\nOutput files saved to: /mnt/user-data/outputs/")
print("="*80)
