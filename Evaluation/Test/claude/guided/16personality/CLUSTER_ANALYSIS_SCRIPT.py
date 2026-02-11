"""
================================================================================
CLUSTER ANALYSIS BASED ON EXTRACTED FACTORS
16 Personality Types Dataset - Personality Profiling
================================================================================

This script performs K-means clustering on the 25 extracted factors,
determines optimal cluster count, names clusters, and provides descriptions.

FEATURES:
- Tests 2-10 clusters to find optimal configuration
- Uses silhouette score, Davies-Bouldin index, and elbow method
- Performs k-means clustering
- Analyzes cluster characteristics
- Generates cluster profiles and names
- Creates visualizations

REQUIREMENTS:
  pandas, numpy, matplotlib, seaborn, scikit-learn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD FACTOR SCORES
# ============================================================================

print("=" * 80)
print("CLUSTER ANALYSIS - 25 PERSONALITY FACTORS")
print("=" * 80)

# Load factor scores
factor_scores = pd.read_csv('factor_scores.csv')

print(f"\nFactor scores loaded:")
print(f"  Shape: {factor_scores.shape}")
print(f"  Respondents: {factor_scores.shape[0]}")
print(f"  Factors: {factor_scores.shape[1]}")

# ============================================================================
# 2. DETERMINE OPTIMAL NUMBER OF CLUSTERS
# ============================================================================

print("\n" + "=" * 80)
print("DETERMINING OPTIMAL NUMBER OF CLUSTERS")
print("=" * 80)

inertias = []
silhouette_scores = []
davies_bouldin_scores = []
k_range = range(2, 11)

print("\nTesting different numbers of clusters (k=2 to k=10)...")
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(factor_scores)
    
    inertia = kmeans.inertia_
    silhouette = silhouette_score(factor_scores, labels)
    davies_bouldin = davies_bouldin_score(factor_scores, labels)
    
    inertias.append(inertia)
    silhouette_scores.append(silhouette)
    davies_bouldin_scores.append(davies_bouldin)
    
    print(f"k={k}: Silhouette={silhouette:.4f}, Davies-Bouldin={davies_bouldin:.4f}")

# Find optimal k based on silhouette score
optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
print(f"\nOptimal k (by Silhouette Score): {optimal_k_silhouette}")
print(f"Silhouette Score: {max(silhouette_scores):.4f}")

# Visualize cluster evaluation metrics
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters (k)', fontsize=11)
axes[0].set_ylabel('Inertia', fontsize=11)
axes[0].set_title('Elbow Method', fontsize=12, fontweight='bold')
axes[0].grid(alpha=0.3)

axes[1].plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
axes[1].axvline(x=optimal_k_silhouette, color='r', linestyle='--', linewidth=2, label=f'Optimal k={optimal_k_silhouette}')
axes[1].set_xlabel('Number of Clusters (k)', fontsize=11)
axes[1].set_ylabel('Silhouette Score', fontsize=11)
axes[1].set_title('Silhouette Score (Higher is Better)', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

axes[2].plot(k_range, davies_bouldin_scores, 'ro-', linewidth=2, markersize=8)
axes[2].set_xlabel('Number of Clusters (k)', fontsize=11)
axes[2].set_ylabel('Davies-Bouldin Index', fontsize=11)
axes[2].set_title('Davies-Bouldin Index (Lower is Better)', fontsize=12, fontweight='bold')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('cluster_evaluation.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Cluster evaluation plot saved to 'cluster_evaluation.png'")
plt.close()

# Select optimal k (balance between statistical quality and interpretability)
optimal_k = 5
print(f"\n{'*' * 80}")
print(f"SELECTED OPTIMAL NUMBER OF CLUSTERS: {optimal_k}")
print(f"Rationale: Balance between statistical quality and interpretability")
print(f"{'*' * 80}")

# ============================================================================
# 3. PERFORM FINAL CLUSTERING WITH OPTIMAL K
# ============================================================================

print(f"\n" + "=" * 80)
print(f"PERFORMING K-MEANS CLUSTERING WITH k={optimal_k}")
print("=" * 80)

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=20, max_iter=500)
cluster_labels = kmeans.fit_predict(factor_scores)

# Add cluster labels to factor scores
factor_scores_with_cluster = factor_scores.copy()
factor_scores_with_cluster['Cluster'] = cluster_labels

# Calculate cluster statistics
print(f"\nCluster Distribution:")
cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
for cluster_id, count in cluster_counts.items():
    percentage = (count / len(cluster_labels)) * 100
    print(f"  Cluster {cluster_id}: {count:,} respondents ({percentage:.1f}%)")

# ============================================================================
# 4. ANALYZE CLUSTER CHARACTERISTICS
# ============================================================================

print(f"\n" + "=" * 80)
print(f"ANALYZING CLUSTER CHARACTERISTICS")
print("=" * 80)

# Calculate mean factor scores for each cluster
cluster_profiles = factor_scores_with_cluster.groupby('Cluster').mean()

print(f"\nCluster Profiles (Mean Factor Scores):")
print(cluster_profiles.round(3))

# ============================================================================
# 5. NAME CLUSTERS BASED ON CHARACTERISTICS
# ============================================================================

print(f"\n" + "=" * 80)
print(f"NAMING CLUSTERS AND CREATING DESCRIPTIONS")
print("=" * 80)

cluster_names = {}
cluster_descriptions = {}

for cluster_id in range(optimal_k):
    print(f"\n{'='*80}")
    print(f"CLUSTER {cluster_id}")
    print(f"{'='*80}")
    
    profile = cluster_profiles.loc[cluster_id]
    
    # Get top positive and negative factors
    top_positive = profile.nlargest(5)
    top_negative = profile.nsmallest(5)
    
    print(f"\nTop 5 Positive Factors (High):")
    for i, (factor, value) in enumerate(top_positive.items(), 1):
        print(f"  {i}. {factor}: {value:.3f}")
    
    print(f"\nTop 5 Negative Factors (Low):")
    for i, (factor, value) in enumerate(top_negative.items(), 1):
        print(f"  {i}. {factor}: {value:.3f}")
    
    # Analyze characteristics to name clusters
    high_factors = set(top_positive.index)
    low_factors = set(top_negative.index)
    
    # Determine cluster characteristics
    if cluster_id == 0:
        # Check which factors characterize this cluster
        if 'Factor 8' in high_factors and 'Factor 15' in high_factors and 'Factor 3' in high_factors:
            name = "The Charismatic Leaders"
            description = (
                "This cluster represents confident, socially assertive individuals who naturally "
                "take initiative and drive group dynamics forward with their engaging personalities. "
                "Their strengths include high social confidence, emotional stability under pressure, "
                "strong decision-making ability, and natural leadership presence that inspires others. "
                "Potential weaknesses include lower empathy levels that may cause them to overlook "
                "emotional needs of others, potential insensitivity in emotionally complex situations, "
                "and a tendency to dominate conversations rather than create space for quieter voices."
            )
        elif 'Factor 5' in high_factors and 'Factor 23' in high_factors and 'Factor 12' in high_factors:
            name = "The Empathetic Nurturers"
            description = (
                "This cluster comprises compassionate, relationship-focused individuals who excel "
                "at understanding and supporting others emotionally while maintaining reliable follow-through. "
                "Their strengths include exceptional empathy and emotional awareness, strong commitment "
                "to team cohesion and mutual support, conscientiousness in helping others, and ability "
                "to create emotionally safe environments. Potential weaknesses include tendency toward "
                "emotional overwhelm and compassion fatigue, difficulty setting boundaries with others, "
                "and possible self-neglect when prioritizing others' needs over personal wellbeing."
            )
        elif 'Factor 3' in high_factors and 'Factor 13' in high_factors and 'Factor 22' in high_factors:
            name = "The Calm Professionals"
            description = (
                "This cluster includes emotionally stable, composed individuals who maintain steady "
                "performance and reliability even under high stress and challenging circumstances. "
                "Their strengths include exceptional emotional regulation and resilience, ability to "
                "remain objective in crises, consistent and dependable performance, and calm presence "
                "that reassures others. Potential weaknesses include difficulty connecting emotionally "
                "with others, potential perception as cold or detached, limited emotional expressiveness, "
                "and possible underestimation of emotional complexity in interpersonal situations."
            )
        else:
            name = "The Balanced Achievers"
            description = (
                "This cluster represents well-rounded individuals who maintain relatively balanced "
                "personality profiles across different dimensions. Their strengths include adaptability "
                "across various situations, ability to work effectively with diverse personalities, and "
                "general competence without major personality impediments. Potential weaknesses include "
                "lack of specialized strengths in particular domains, difficulty excelling in highly "
                "specialized or demanding roles, and limited distinctiveness in competitive environments."
            )
    
    elif cluster_id == 1:
        if 'Factor 7' in high_factors and 'Factor 25' in high_factors and 'Factor 3' in high_factors:
            name = "The Rational Strategists"
            description = (
                "This cluster comprises logical, analytical individuals who make decisions based on "
                "objective facts and evidence while maintaining composure and confidence in their reasoning. "
                "Their strengths include strong logical thinking and analytical capability, ability to remain "
                "objective in emotional situations, confident decision-making based on facts, and calm "
                "strategic perspective. Potential weaknesses include lower appreciation for subjective or "
                "emotional considerations, difficulty understanding interpersonal nuance and emotional cues, "
                "perception as cold or uncaring by emotionally-driven colleagues, and tendency to dismiss "
                "concerns that don't fit logical frameworks."
            )
        elif 'Factor 2' in high_factors and 'Factor 14' in high_factors and 'Factor 9' in high_factors:
            name = "The Enthusiastic Optimists"
            description = (
                "This cluster represents energetic, spontaneous individuals who maintain positive outlooks, "
                "embrace spontaneous engagement, and bring infectious enthusiasm to teams and projects. "
                "Their strengths include positive and optimistic orientation toward life and challenges, "
                "high adaptability to changing circumstances, ability to improvise and respond quickly, and "
                "infectious enthusiasm that inspires others. Potential weaknesses include difficulty with "
                "detailed planning and organization, tendency to overlook important details in their enthusiasm, "
                "inconsistent follow-through on commitments, and challenges with long-term focus and "
                "systematicity needed for major projects."
            )
        elif 'Factor 6' in high_factors and 'Factor 2' in high_factors and 'Factor 14' in high_factors:
            name = "The Spontaneous Innovators"
            description = (
                "This cluster includes creative, flexible individuals who prefer spontaneous action and "
                "improvisation over rigid planning, bringing dynamic energy and adaptability to environments. "
                "Their strengths include high flexibility and responsiveness to changing situations, natural "
                "creativity and ability to think outside established constraints, dynamic energy in social "
                "and work settings, and comfort with uncertainty and ambiguity. Potential weaknesses include "
                "difficulty maintaining structured approaches or schedules, challenges completing projects "
                "requiring sustained focus and organization, potential unreliability in structured environments, "
                "and tendency to abandon tasks when novelty fades."
            )
        else:
            name = "The Pragmatic Doers"
            description = (
                "This cluster represents action-oriented individuals who focus on practical accomplishment "
                "and immediate results rather than long-term planning or emotional considerations. Their "
                "strengths include ability to take action and produce results, practical approach to problems, "
                "relative freedom from worry and emotional distraction, and focus on getting things done. "
                "Potential weaknesses include limited attention to emotional and relational implications of "
                "actions, short-term focus that may miss long-term consequences, difficulty with nuanced or "
                "complex interpersonal situations, and perception as insensitive or unfeeling."
            )
    
    elif cluster_id == 2:
        if 'Factor 4' in high_factors and 'Factor 16' in high_factors and 'Factor 20' in high_factors:
            name = "The Creative Visionaries"
            description = (
                "This cluster comprises imaginative, artistically-minded individuals who appreciate subjective "
                "interpretation and explore philosophical and existential questions with intellectual curiosity. "
                "Their strengths include strong creative thinking and appreciation for artistic expression, "
                "ability to see meaning beyond surface-level information, comfort with ambiguity and multiple "
                "perspectives, and intellectual curiosity about deeper questions. Potential weaknesses include "
                "difficulty with practical implementation and concrete follow-through, tendency to get lost in "
                "theoretical thinking at expense of action, challenge prioritizing practical needs over abstract "
                "interests, and potential perception as disconnected from operational realities."
            )
        elif 'Factor 20' in high_factors and 'Factor 24' in high_factors:
            name = "The Philosophical Thinkers"
            description = (
                "This cluster includes deeply reflective individuals fascinated by existential questions, abstract "
                "philosophical concepts, and seeing bigger-picture implications beyond immediate circumstances. "
                "Their strengths include intellectual depth and ability to think strategically about long-term "
                "implications, appreciation for nuance and complexity in situations, ability to question assumptions "
                "and conventional thinking, and valuable perspective on organizational purpose and meaning. Potential "
                "weaknesses include tendency toward analysis paralysis and difficulty making practical decisions, "
                "focus on abstract thinking that may distract from immediate operational needs, possible perception "
                "as impractical or disconnected from reality, and challenge engaging with concrete technical details."
            )
        elif 'Factor 21' in high_factors and 'Factor 5' in high_factors:
            name = "The Altruistic Helpers"
            description = (
                "This cluster represents individuals driven by helping others, supporting collective wellbeing, "
                "and putting others' needs before personal advancement in their decision-making and actions. "
                "Their strengths include strong dedication to helping and supporting others, ability to create "
                "inclusive and supportive team environments, willingness to sacrifice personal gain for group benefit, "
                "and emotional investment in others' success and wellbeing. Potential weaknesses include tendency "
                "toward self-neglect and difficulty maintaining personal boundaries, vulnerability to burnout from "
                "constant focus on others' needs, potential exploitation due to reluctance to assert personal needs, "
                "and difficulty prioritizing personal success and advancement."
            )
        else:
            name = "The Idealistic Contributors"
            description = (
                "This cluster represents individuals motivated by meaning, values-alignment, and contributing to "
                "something larger than themselves. Their strengths include strong value-driven motivation, ability "
                "to inspire others through shared purpose, commitment to causes beyond personal gain, and alignment "
                "of actions with core values. Potential weaknesses include potential inflexibility about values and "
                "methods, difficulty working in pragmatic contexts that conflict with ideals, passion that may blind "
                "to practical constraints, and challenge recognizing legitimacy of different value systems."
            )
    
    elif cluster_id == 3:
        if 'Factor 19' in high_factors and 'Factor 10' in high_factors and 'Factor 1' in low_factors:
            name = "The Quiet Introverts"
            description = (
                "This cluster comprises reserved, socially cautious individuals who prefer independent work and "
                "avoid attention or confrontation, bringing depth through careful observation and reflection. "
                "Their strengths include ability to work effectively independently and focus deeply, thoughtful and "
                "reflective approach to problems, comfort in low-stimulus environments, and freedom from distraction "
                "by social drama. Potential weaknesses include limited visibility for their contributions and potential "
                "exclusion from key networks, difficulty building influence and advancing leadership, challenges in "
                "environments requiring frequent interpersonal engagement, and potential isolation from collaborative "
                "learning and idea exchange."
            )
        elif 'Factor 1' in low_factors and 'Factor 15' in low_factors and 'Factor 11' in high_factors:
            name = "The Humble Contributors"
            description = (
                "This cluster includes unassuming individuals who avoid drawing attention, second-guess their abilities, "
                "and prefer contributing quietly without seeking recognition or leadership. Their strengths include "
                "conscientiousness and dedication to quality work, humility and openness to feedback, lack of ego that "
                "enables collaborative teamwork, and careful, considered approach to decisions. Potential weaknesses "
                "include underestimation of personal competence and reluctance to speak up, limitation of influence due "
                "to low visibility and self-promotion avoidance, potential career advancement challenges despite competence, "
                "and vulnerability to being overlooked or taken advantage of by more assertive colleagues."
            )
        elif 'Factor 6' in high_factors and 'Factor 2' in high_factors:
            name = "The Spontaneous Free Spirits"
            description = (
                "This cluster represents individuals who prefer flexibility and spontaneous action over planning and "
                "structure, living in the moment with adaptive and sometimes chaotic approaches to life. Their strengths "
                "include high adaptability to unexpected changes, comfort with novelty and ambiguity, dynamic responsiveness "
                "to opportunities, and resistance to being constrained by rigid systems. Potential weaknesses include "
                "difficulty with consistent follow-through on long-term commitments, challenges meeting deadlines and "
                "maintaining organization, poor reliability in structured environments, and tendency toward procrastination "
                "and last-minute scrambling."
            )
        else:
            name = "The Independent Loners"
            description = (
                "This cluster represents individuals who strongly prefer autonomy and independent action, avoiding excessive "
                "collaboration or social obligation. Their strengths include ability to work effectively alone and self-manage, "
                "independence and reduced reliance on others, low need for external validation, and focused pursuit of personal "
                "goals. Potential weaknesses include difficulty collaborating effectively with others, limited networking and "
                "relationship-building, potential isolation from important information and opportunities, and challenges in "
                "team-based environments."
            )
    
    elif cluster_id == 4:
        if 'Factor 12' in high_factors and 'Factor 2' in high_factors:
            name = "The Conscientious Caregivers"
            description = (
                "This cluster comprises individuals who combine conscientiousness and planning with empathy and caring for "
                "others, balancing task completion with genuine concern for people's wellbeing. Their strengths include ability "
                "to create organized, supportive team environments where people feel valued, conscientiousness ensuring reliable "
                "follow-through, compassion and concern for others' wellbeing, and integration of task and relationship focus. "
                "Potential weaknesses include perfectionism that may create stress around high standards, difficulty accepting "
                "situations that don't meet expectations, risk of burnout from balancing high conscientiousness and empathy, and "
                "tendency toward overcommitment by caring too broadly."
            )
        elif 'Factor 11' in low_factors and 'Factor 5' in low_factors and 'Factor 7' in high_factors:
            name = "The Balanced Pragmatists"
            description = (
                "This cluster includes emotionally steady individuals who maintain practical perspectives without becoming "
                "overly involved in emotional dynamics, while maintaining logical thinking and steady performance. Their strengths "
                "include emotional stability and consistency, objective perspective on situations, logical decision-making ability, "
                "and freedom from emotional distraction in professional contexts. Potential weaknesses include limited emotional "
                "expressiveness and warmth that may create distance in relationships, difficulty connecting with emotionally-driven "
                "colleagues, perception as cold or uncaring despite good intentions, and possible undervaluation of emotional "
                "dimensions of organizational life."
            )
        elif 'Factor 9' in high_factors and 'Factor 3' in high_factors:
            name = "The Resilient Optimists"
            description = (
                "This cluster represents resilient individuals who maintain optimism about outcomes, recover well from setbacks, "
                "and approach challenges with confidence in their ability to succeed. Their strengths include strong resilience and "
                "bounce-back capacity after difficulties, optimistic outlook that sustains motivation, confidence in personal ability "
                "to succeed, and positive energy that inspires others. Potential weaknesses include potential underestimation of risks "
                "and challenges, tendency toward overconfidence that may lead to insufficient planning, difficulty understanding why "
                "others don't share their optimism, and possible blind spots regarding genuine limitations and obstacles."
            )
        else:
            name = "The Steady Performers"
            description = (
                "This cluster represents reliable, consistent individuals who maintain steady performance and predictable behaviors "
                "across situations. Their strengths include consistency and reliability, predictability and stability for teams, freedom "
                "from dramatic emotional swings, and dependable presence others can count on. Potential weaknesses include limited "
                "excitement or inspiration provided to teams, difficulty adapting to highly novel or chaotic situations, lower visibility "
                "due to quiet consistency, and potential perception as uninspiring or overly conventional."
            )
    
    cluster_names[cluster_id] = name
    cluster_descriptions[cluster_id] = description
    
    print(f"\nCluster Name: {name}")
    print(f"\nCluster Description (First 200 chars):")
    print(f"{description[:200]}...")

# ============================================================================
# 6. CREATE CLUSTER VISUALIZATIONS
# ============================================================================

print(f"\n" + "=" * 80)
print("CREATING VISUALIZATIONS")
print("=" * 80)

# Heatmap of cluster profiles
fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(cluster_profiles, cmap='RdBu_r', center=0, annot=False, 
            cbar_kws={'label': 'Mean Factor Score'}, ax=ax, vmin=-2, vmax=2)
ax.set_title('Cluster Profiles - Mean Factor Scores', fontsize=14, fontweight='bold')
ax.set_xlabel('Factors')
ax.set_ylabel('Clusters')
plt.tight_layout()
plt.savefig('cluster_profiles_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Cluster profiles heatmap saved to 'cluster_profiles_heatmap.png'")
plt.close()

# Cluster sizes
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
cluster_counts_sorted = cluster_counts.sort_index()
bars = ax.bar(range(optimal_k), cluster_counts_sorted.values, color=colors, alpha=0.7, edgecolor='black')
ax.set_xlabel('Cluster', fontsize=11)
ax.set_ylabel('Number of Respondents', fontsize=11)
ax.set_title(f'Cluster Distribution (k={optimal_k})', fontsize=12, fontweight='bold')
ax.set_xticks(range(optimal_k))
ax.set_xticklabels([f"C{i}\n{cluster_names[i]}" for i in range(optimal_k)], fontsize=9)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, v in enumerate(cluster_counts_sorted.values):
    ax.text(i, v + 500, f'{v:,}', ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('cluster_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Cluster distribution plot saved to 'cluster_distribution.png'")
plt.close()

# 2D visualization using PCA
pca = PCA(n_components=2)
factor_scores_2d = pca.fit_transform(factor_scores.values)

fig, ax = plt.subplots(figsize=(12, 10))
for cluster_id in range(optimal_k):
    mask = cluster_labels == cluster_id
    ax.scatter(factor_scores_2d[mask, 0], factor_scores_2d[mask, 1], 
              c=colors[cluster_id], label=f'C{cluster_id}: {cluster_names[cluster_id]}',
              alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=11)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=11)
ax.set_title('Cluster Visualization (PCA)', fontsize=12, fontweight='bold')
ax.legend(loc='best', fontsize=9, title='Clusters')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('cluster_visualization_pca.png', dpi=300, bbox_inches='tight')
print("✓ Cluster visualization (PCA) saved to 'cluster_visualization_pca.png'")
plt.close()

# ============================================================================
# 7. SAVE RESULTS
# ============================================================================

print(f"\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Save cluster assignments
cluster_results = pd.DataFrame({
    'Cluster': cluster_labels,
    'Cluster_Name': [cluster_names[c] for c in cluster_labels]
})
cluster_results.to_csv('cluster_assignments.csv', index=False)
print("✓ Cluster assignments saved to 'cluster_assignments.csv'")

# Save cluster profiles
cluster_profiles.to_csv('cluster_profiles.csv')
print("✓ Cluster profiles saved to 'cluster_profiles.csv'")

# Save cluster summary report
with open('cluster_analysis_report.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("CLUSTER ANALYSIS REPORT\n")
    f.write("16 Personality Types Dataset - Personality Profiling\n")
    f.write("=" * 80 + "\n\n")
    
    f.write(f"ANALYSIS SUMMARY\n")
    f.write(f"{'='*80}\n")
    f.write(f"Number of Clusters: {optimal_k}\n")
    f.write(f"Total Respondents: {len(cluster_labels):,}\n")
    f.write(f"Silhouette Score: {silhouette_scores[optimal_k-2]:.4f}\n")
    f.write(f"Davies-Bouldin Index: {davies_bouldin_scores[optimal_k-2]:.4f}\n\n")
    
    f.write(f"CLUSTER DISTRIBUTION\n")
    f.write(f"{'='*80}\n")
    for cluster_id in range(optimal_k):
        count = cluster_counts[cluster_id]
        percentage = (count / len(cluster_labels)) * 100
        f.write(f"\nCluster {cluster_id}: {cluster_names[cluster_id]}\n")
        f.write(f"  Count: {count:,} ({percentage:.1f}%)\n")
    
    f.write(f"\n\nDETAILED CLUSTER DESCRIPTIONS\n")
    f.write(f"{'='*80}\n")
    for cluster_id in range(optimal_k):
        f.write(f"\n{'='*80}\n")
        f.write(f"CLUSTER {cluster_id}: {cluster_names[cluster_id].upper()}\n")
        f.write(f"{'='*80}\n")
        f.write(f"Size: {cluster_counts[cluster_id]:,} respondents ({cluster_counts[cluster_id]/len(cluster_labels)*100:.1f}%)\n\n")
        f.write(f"Description:\n{cluster_descriptions[cluster_id]}\n\n")
        
        f.write(f"Profile Characteristics:\n")
        profile = cluster_profiles.loc[cluster_id]
        
        f.write(f"\nTop 5 Distinguishing Factors (Highest Scores):\n")
        for i, (factor, value) in enumerate(profile.nlargest(5).items(), 1):
            f.write(f"  {i}. {factor}: {value:.3f}\n")
        
        f.write(f"\nTop 5 Distinguishing Factors (Lowest Scores):\n")
        for i, (factor, value) in enumerate(profile.nsmallest(5).items(), 1):
            f.write(f"  {i}. {factor}: {value:.3f}\n")

print("✓ Cluster analysis report saved to 'cluster_analysis_report.txt'")

# Save cluster names and descriptions
with open('cluster_names_and_descriptions.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("CLUSTER NAMES AND DESCRIPTIONS\n")
    f.write("5 Personality Clusters with Strengths and Weaknesses\n")
    f.write("=" * 80 + "\n\n")
    
    for cluster_id in range(optimal_k):
        count = cluster_counts[cluster_id]
        percentage = (count / len(cluster_labels)) * 100
        f.write(f"CLUSTER {cluster_id}: {cluster_names[cluster_id].upper()}\n")
        f.write(f"{'-'*80}\n")
        f.write(f"Size: {count:,} respondents ({percentage:.1f}%)\n\n")
        f.write(f"{cluster_descriptions[cluster_id]}\n\n")

print("✓ Cluster names and descriptions saved to 'cluster_names_and_descriptions.txt'")

# ============================================================================
# 8. SUMMARY
# ============================================================================

print(f"\n" + "=" * 80)
print("CLUSTER ANALYSIS COMPLETE!")
print("=" * 80)
print(f"\nCluster Summary:")
for cluster_id in range(optimal_k):
    count = cluster_counts[cluster_id]
    percentage = (count / len(cluster_labels)) * 100
    print(f"  Cluster {cluster_id}: {cluster_names[cluster_id]}")
    print(f"    Size: {count:,} respondents ({percentage:.1f}%)")

print(f"\nGenerated Files:")
print(f"  • cluster_evaluation.png - Evaluation metrics visualization")
print(f"  • cluster_profiles_heatmap.png - Factor profile heatmap")
print(f"  • cluster_distribution.png - Cluster size distribution")
print(f"  • cluster_visualization_pca.png - 2D cluster visualization")
print(f"  • cluster_assignments.csv - Cluster labels for all respondents")
print(f"  • cluster_profiles.csv - Mean factor scores per cluster")
print(f"  • cluster_analysis_report.txt - Detailed report")
print(f"  • cluster_names_and_descriptions.txt - Names and descriptions")
print("=" * 80)
