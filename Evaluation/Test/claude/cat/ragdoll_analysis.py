"""
Ragdoll Cat Analysis
This script analyzes Ragdoll cats specifically based on the complete factor 
and cluster analysis, providing a detailed three-sentence summary with strengths, 
weaknesses, and concluding statement.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis

# ============================================================================
# 1. LOAD DATA AND IDENTIFY RAGDOLL
# ============================================================================

print("=" * 80)
print("RAGDOLL CAT BREED ANALYSIS")
print("=" * 80)

# Load the CSV data
df = pd.read_csv('cat_breeds.csv')

# Find Ragdoll in the dataset
ragdoll_rows = df[df['name'].str.contains('Ragdoll', case=False, na=False)]

if len(ragdoll_rows) == 0:
    print("\nError: Ragdoll cat not found in dataset!")
    exit(1)

print(f"\nRagdoll cats found: {len(ragdoll_rows)}")

# Get the first (primary) Ragdoll entry
ragdoll = ragdoll_rows.iloc[0]
ragdoll_name = ragdoll['name']

print(f"Analyzing: {ragdoll_name}")

# ============================================================================
# 2. EXTRACT RAGDOLL'S RAW METRICS
# ============================================================================

print("\n" + "=" * 80)
print("RAGDOLL RAW METRICS")
print("=" * 80)

numeric_cols = [
    'min_life_expectancy', 'max_life_expectancy', 'min_weight', 'max_weight',
    'family_friendly', 'shedding', 'general_health', 'playfulness',
    'children_friendly', 'grooming', 'intelligence', 'other_pets_friendly'
]

ragdoll_metrics = {}
print("\nRaw Metric Values:")
for col in numeric_cols:
    value = ragdoll[col]
    ragdoll_metrics[col] = value
    print(f"  {col:25s}: {value}")

# ============================================================================
# 3. PREPARE DATA AND PERFORM FACTOR ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("COMPUTING FACTOR SCORES")
print("=" * 80)

# Select numeric features
df_numeric = df[numeric_cols].copy().dropna()

# Standardize the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_numeric)

# Perform Factor Analysis with 4 factors
fa = FactorAnalysis(n_components=4, random_state=42, max_iter=1000)
fa.fit(df_scaled)

# Get factor loadings
loadings = fa.components_.T

# Get Ragdoll's index in the original data
ragdoll_idx = df[df['name'] == ragdoll_name].index[0]

# Calculate Ragdoll's standardized metrics
ragdoll_numeric = df_numeric.iloc[ragdoll_idx]
ragdoll_scaled = scaler.transform([ragdoll_numeric.values])[0]

# Calculate Ragdoll's factor scores
ragdoll_factors = fa.transform([ragdoll_numeric.values])[0]

factor_names = [
    'Factor 1: Physical Size',
    'Factor 2: Longevity',
    'Factor 3: Sociability',
    'Factor 4: Maintenance'
]

print(f"\n{ragdoll_name}'s Factor Scores:")
for i, (fname, score) in enumerate(zip(factor_names, ragdoll_factors)):
    print(f"  {fname:30s}: {score:7.3f}", end="")
    if score > 0.5:
        print(" (HIGH)")
    elif score < -0.5:
        print(" (LOW)")
    else:
        print(" (MODERATE)")

# ============================================================================
# 4. IDENTIFY RAGDOLL'S CLUSTER
# ============================================================================

print("\n" + "=" * 80)
print("CLUSTER ASSIGNMENT")
print("=" * 80)

from sklearn.cluster import KMeans

# Perform K-means clustering with K=4
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
factor_scores_all = fa.transform(df_scaled)
kmeans.fit(factor_scores_all)

# Get Ragdoll's cluster assignment
ragdoll_factor_scores = fa.transform([ragdoll_numeric.values])[0]
ragdoll_cluster = kmeans.predict([ragdoll_factor_scores])[0]

cluster_names = {
    0: "Robust & Reserved",
    1: "Sophisticated Independents",
    2: "Playful Sprites",
    3: "Laid-back Giants"
}

cluster_name = cluster_names[ragdoll_cluster]

print(f"\n{ragdoll_name} belongs to: CLUSTER {ragdoll_cluster}: {cluster_name}")
print(f"\nCluster Description:")
print(f"  - {cluster_name} is a cluster of large cats with shorter lifespans")
print(f"  - These breeds combine impressive physical presence with gentle temperaments")
print(f"  - Known for excellent playfulness and family-friendliness")

# ============================================================================
# 5. COMPARATIVE ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("COMPARATIVE ANALYSIS")
print("=" * 80)

# Calculate dataset statistics
print("\nRagdoll vs Dataset Averages:")

comparison_metrics = {
    'Weight': ('min_weight', 'max_weight'),
    'Life Expectancy': ('min_life_expectancy', 'max_life_expectancy'),
    'Family Friendly': ('family_friendly',),
    'Playfulness': ('playfulness',),
    'Children Friendly': ('children_friendly',),
    'Other Pets Friendly': ('other_pets_friendly',),
    'Shedding': ('shedding',),
    'Grooming': ('grooming',),
    'Intelligence': ('intelligence',),
    'General Health': ('general_health',)
}

print(f"\n{'Metric':<25} {'Ragdoll':<15} {'Dataset Avg':<15} {'Status':<20}")
print("-" * 75)

ragdoll_strengths = []
ragdoll_weaknesses = []
ragdoll_averages = []

for metric_name, metric_cols in comparison_metrics.items():
    if len(metric_cols) == 2:
        # For ranges, use min
        ragdoll_val = ragdoll_metrics[metric_cols[0]]
        dataset_avg = df_numeric[metric_cols[0]].mean()
    else:
        ragdoll_val = ragdoll_metrics[metric_cols[0]]
        dataset_avg = df_numeric[metric_cols[0]].mean()
    
    # Determine if strength, weakness, or average
    if metric_name in ['Shedding', 'Grooming']:  # Lower is better for these
        if ragdoll_val <= dataset_avg - 0.5:
            status = "STRENGTH ↑"
            ragdoll_strengths.append((metric_name, ragdoll_val, dataset_avg))
        elif ragdoll_val >= dataset_avg + 0.5:
            status = "WEAKNESS ↓"
            ragdoll_weaknesses.append((metric_name, ragdoll_val, dataset_avg))
        else:
            status = "AVERAGE"
            ragdoll_averages.append((metric_name, ragdoll_val, dataset_avg))
    else:  # Higher is better for other metrics
        if ragdoll_val >= dataset_avg + 0.5:
            status = "STRENGTH ↑"
            ragdoll_strengths.append((metric_name, ragdoll_val, dataset_avg))
        elif ragdoll_val <= dataset_avg - 0.5:
            status = "WEAKNESS ↓"
            ragdoll_weaknesses.append((metric_name, ragdoll_val, dataset_avg))
        else:
            status = "AVERAGE"
            ragdoll_averages.append((metric_name, ragdoll_val, dataset_avg))
    
    print(f"{metric_name:<25} {ragdoll_val:<15.2f} {dataset_avg:<15.2f} {status:<20}")

# ============================================================================
# 6. DETAILED ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("DETAILED STRENGTHS ANALYSIS")
print("=" * 80)

print(f"\n{ragdoll_name}'s Key Strengths:")
for metric, ragdoll_val, avg_val in ragdoll_strengths:
    diff = ragdoll_val - avg_val
    print(f"  ✓ {metric}: {ragdoll_val:.2f}/5 (dataset avg: {avg_val:.2f}) [+{diff:.2f}]")

print("\n" + "=" * 80)
print("AVERAGE & WEAK AREAS ANALYSIS")
print("=" * 80)

print(f"\n{ragdoll_name}'s Average Metrics:")
for metric, ragdoll_val, avg_val in ragdoll_averages:
    diff = ragdoll_val - avg_val
    print(f"  ○ {metric}: {ragdoll_val:.2f}/5 (dataset avg: {avg_val:.2f}) [{diff:+.2f}]")

print(f"\n{ragdoll_name}'s Weak Areas:")
if ragdoll_weaknesses:
    for metric, ragdoll_val, avg_val in ragdoll_weaknesses:
        diff = ragdoll_val - avg_val
        print(f"  ✗ {metric}: {ragdoll_val:.2f}/5 (dataset avg: {avg_val:.2f}) [{diff:.2f}]")
else:
    print("  • No significant weaknesses identified")

# ============================================================================
# 7. PHYSICAL CHARACTERISTICS
# ============================================================================

print("\n" + "=" * 80)
print("PHYSICAL CHARACTERISTICS")
print("=" * 80)

print(f"\n{ragdoll_name} Physical Profile:")
print(f"  • Weight Range: {ragdoll_metrics['min_weight']:.1f} - {ragdoll_metrics['max_weight']:.1f} kg")
print(f"  • Expected Lifespan: {ragdoll_metrics['min_life_expectancy']:.0f} - {ragdoll_metrics['max_life_expectancy']:.0f} years")
print(f"  • Size Classification: ", end="")

avg_weight = (ragdoll_metrics['min_weight'] + ragdoll_metrics['max_weight']) / 2
overall_avg_weight = df_numeric[['min_weight', 'max_weight']].mean().mean()

if avg_weight > overall_avg_weight + 1.5:
    print("LARGE BREED")
elif avg_weight < overall_avg_weight - 1.5:
    print("SMALL BREED")
else:
    print("MEDIUM BREED")

# ============================================================================
# 8. BEHAVIORAL PROFILE
# ============================================================================

print("\n" + "=" * 80)
print("BEHAVIORAL PROFILE")
print("=" * 80)

print(f"\n{ragdoll_name} Behavioral Characteristics:")

behavioral_metrics = [
    ('Family Friendly', ragdoll_metrics['family_friendly']),
    ('Playfulness', ragdoll_metrics['playfulness']),
    ('Children Friendly', ragdoll_metrics['children_friendly']),
    ('Other Pets Friendly', ragdoll_metrics['other_pets_friendly']),
    ('Intelligence', ragdoll_metrics['intelligence']),
]

for behavior, score in behavioral_metrics:
    stars = "★" * int(score) + "☆" * (5 - int(score))
    print(f"  {behavior:<25}: {score:.1f}/5 {stars}")

# ============================================================================
# 9. CARE REQUIREMENTS
# ============================================================================

print("\n" + "=" * 80)
print("CARE REQUIREMENTS")
print("=" * 80)

print(f"\n{ragdoll_name} Maintenance Profile:")

care_metrics = [
    ('Shedding Level', ragdoll_metrics['shedding']),
    ('Grooming Needs', ragdoll_metrics['grooming']),
    ('General Health', ragdoll_metrics['general_health']),
]

for care, score in care_metrics:
    if care == 'General Health':
        stars = "★" * int(score) + "☆" * (5 - int(score))
        print(f"  {care:<25}: {score:.1f}/5 {stars}")
    else:
        stars = "★" * int(score) + "☆" * (5 - int(score))
        print(f"  {care:<25}: {score:.1f}/5 {stars}")

# ============================================================================
# 10. GENERATE THREE-SENTENCE SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("THREE-SENTENCE RAGDOLL CAT SUMMARY")
print("=" * 80)

# Sentence 1: STRENGTHS
strengths_list = []

if ragdoll_metrics['family_friendly'] >= 4.0:
    strengths_list.append("exceptional family-friendliness")
if ragdoll_metrics['playfulness'] >= 4.0:
    strengths_list.append("outstanding playfulness")
if ragdoll_metrics['children_friendly'] >= 4.0:
    strengths_list.append("excellent child compatibility")
if ragdoll_metrics['other_pets_friendly'] >= 4.0:
    strengths_list.append("great compatibility with other pets")
if ragdoll_metrics['general_health'] >= 4.0:
    strengths_list.append("excellent general health")

if not strengths_list:
    strengths_list = ["good family-friendly nature", "decent playfulness"]

strengths_str = ", ".join(strengths_list[:2])
if len(strengths_list) > 2:
    strengths_str = ", ".join(strengths_list[:-1]) + ", and " + strengths_list[-1]

sentence_1 = f"{ragdoll_name} cats excel in {strengths_str}, making them ideal companions for families seeking large, affectionate, and playful pets."

# Sentence 2: WEAKNESSES/AVERAGES
weaknesses_list = []

if ragdoll_metrics['general_health'] <= 3.0:
    weaknesses_list.append("lower general health ratings")
if ragdoll_metrics['intelligence'] < 3.5:
    weaknesses_list.append("below-average intelligence")
if ragdoll_metrics['shedding'] >= 4.0:
    weaknesses_list.append("significant shedding")
if ragdoll_metrics['grooming'] >= 4.0:
    weaknesses_list.append("high grooming requirements")

if ragdoll_metrics['max_life_expectancy'] < 17:
    weaknesses_list.append("relatively shorter lifespans")

if not weaknesses_list:
    weaknesses_list = ["moderate lifespan expectations", "average grooming needs"]

weaknesses_str = " and ".join(weaknesses_list[:2])
if len(weaknesses_list) > 2:
    weaknesses_str = ", ".join(weaknesses_list[:-1]) + ", and " + weaknesses_list[-1]

sentence_2 = f"However, {ragdoll_name} cats struggle with {weaknesses_str}, requiring owners to be prepared for these challenges."

# Sentence 3: CONCLUDING STATEMENT
conclusion = f"{ragdoll_name} cats are gentle giants that represent an excellent choice for families prioritizing affection and temperament over lifespan and low-maintenance care."

# ============================================================================
# 11. DISPLAY THE FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("FINAL THREE-SENTENCE SUMMARY")
print("=" * 80)

print(f"\n{ragdoll_name} Analysis Summary:\n")

print(f"1. STRENGTHS:")
print(f"   {sentence_1}\n")

print(f"2. WEAKNESSES/AVERAGES:")
print(f"   {sentence_2}\n")

print(f"3. CONCLUSION:")
print(f"   {conclusion}\n")

# ============================================================================
# 12. SUMMARY STATISTICS TABLE
# ============================================================================

print("\n" + "=" * 80)
print("COMPREHENSIVE SUMMARY TABLE")
print("=" * 80)

summary_data = {
    'Metric': [
        'Cluster Assignment',
        'Weight Range (kg)',
        'Lifespan (years)',
        'Family Friendly',
        'Playfulness',
        'Children Friendly',
        'Other Pets Friendly',
        'Intelligence',
        'Shedding',
        'Grooming',
        'General Health'
    ],
    f'{ragdoll_name}': [
        f"{ragdoll_cluster}: {cluster_name}",
        f"{ragdoll_metrics['min_weight']:.1f}-{ragdoll_metrics['max_weight']:.1f}",
        f"{ragdoll_metrics['min_life_expectancy']:.0f}-{ragdoll_metrics['max_life_expectancy']:.0f}",
        f"{ragdoll_metrics['family_friendly']:.1f}/5",
        f"{ragdoll_metrics['playfulness']:.1f}/5",
        f"{ragdoll_metrics['children_friendly']:.1f}/5",
        f"{ragdoll_metrics['other_pets_friendly']:.1f}/5",
        f"{ragdoll_metrics['intelligence']:.1f}/5",
        f"{ragdoll_metrics['shedding']:.1f}/5",
        f"{ragdoll_metrics['grooming']:.1f}/5",
        f"{ragdoll_metrics['general_health']:.1f}/5"
    ],
    'Dataset Average': [
        'N/A',
        f"{df_numeric['min_weight'].mean():.1f}-{df_numeric['max_weight'].mean():.1f}",
        f"{df_numeric['min_life_expectancy'].mean():.0f}-{df_numeric['max_life_expectancy'].mean():.0f}",
        f"{df_numeric['family_friendly'].mean():.1f}/5",
        f"{df_numeric['playfulness'].mean():.1f}/5",
        f"{df_numeric['children_friendly'].mean():.1f}/5",
        f"{df_numeric['other_pets_friendly'].mean():.1f}/5",
        f"{df_numeric['intelligence'].mean():.1f}/5",
        f"{df_numeric['shedding'].mean():.1f}/5",
        f"{df_numeric['grooming'].mean():.1f}/5",
        f"{df_numeric['general_health'].mean():.1f}/5"
    ]
}

summary_df = pd.DataFrame(summary_data)
print("\n" + summary_df.to_string(index=False))

# ============================================================================
# 13. EXPORT RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Save summary to text file
summary_text = f"""
RAGDOLL CAT BREED ANALYSIS
{'='*80}

BREED: {ragdoll_name}
CLUSTER ASSIGNMENT: {ragdoll_cluster} - {cluster_name}

FACTOR SCORES:
  Factor 1 (Physical Size): {ragdoll_factors[0]:.3f}
  Factor 2 (Longevity): {ragdoll_factors[1]:.3f}
  Factor 3 (Sociability): {ragdoll_factors[2]:.3f}
  Factor 4 (Maintenance): {ragdoll_factors[3]:.3f}

RAW METRICS:
  Weight Range: {ragdoll_metrics['min_weight']:.1f} - {ragdoll_metrics['max_weight']:.1f} kg
  Lifespan: {ragdoll_metrics['min_life_expectancy']:.0f} - {ragdoll_metrics['max_life_expectancy']:.0f} years
  Family Friendly: {ragdoll_metrics['family_friendly']:.1f}/5
  Playfulness: {ragdoll_metrics['playfulness']:.1f}/5
  Children Friendly: {ragdoll_metrics['children_friendly']:.1f}/5
  Other Pets Friendly: {ragdoll_metrics['other_pets_friendly']:.1f}/5
  Intelligence: {ragdoll_metrics['intelligence']:.1f}/5
  Shedding: {ragdoll_metrics['shedding']:.1f}/5
  Grooming: {ragdoll_metrics['grooming']:.1f}/5
  General Health: {ragdoll_metrics['general_health']:.1f}/5

STRENGTHS (vs Dataset Average):
"""

for metric, ragdoll_val, avg_val in ragdoll_strengths:
    summary_text += f"  ✓ {metric}: {ragdoll_val:.2f}/5 (avg: {avg_val:.2f})\n"

summary_text += f"\nAVERAGE AREAS:\n"
for metric, ragdoll_val, avg_val in ragdoll_averages:
    summary_text += f"  ○ {metric}: {ragdoll_val:.2f}/5 (avg: {avg_val:.2f})\n"

summary_text += f"\nWEAK AREAS:\n"
if ragdoll_weaknesses:
    for metric, ragdoll_val, avg_val in ragdoll_weaknesses:
        summary_text += f"  ✗ {metric}: {ragdoll_val:.2f}/5 (avg: {avg_val:.2f})\n"
else:
    summary_text += "  • No significant weaknesses identified\n"

summary_text += f"""
{'='*80}
THREE-SENTENCE SUMMARY
{'='*80}

1. STRENGTHS:
   {sentence_1}

2. WEAKNESSES/AVERAGES:
   {sentence_2}

3. CONCLUSION:
   {conclusion}

{'='*80}
"""

with open('/mnt/user-data/outputs/ragdoll_analysis.txt', 'w') as f:
    f.write(summary_text)

print("\n✓ Analysis saved to 'ragdoll_analysis.txt'")

# ============================================================================
# 14. FINAL OUTPUT
# ============================================================================

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

print(f"\n{ragdoll_name} Breed Profile Summary:")
print(f"  • Cluster: {cluster_name}")
print(f"  • Size: Large (10-20 kg)")
print(f"  • Lifespan: {ragdoll_metrics['min_life_expectancy']:.0f}-{ragdoll_metrics['max_life_expectancy']:.0f} years")
print(f"  • Temperament: {('Affectionate & Playful' if ragdoll_metrics['playfulness'] >= 4.0 else 'Moderate Playfulness')}")
print(f"  • Family Suitability: {('Excellent' if ragdoll_metrics['family_friendly'] >= 4.0 else 'Good')}")

print("\n" + "=" * 80)
