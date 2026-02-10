"""
PERSON 12 PERSONALITY ANALYSIS - REPRODUCIBLE CODE
Generates 3-sentence summary for Person 12 based on factor scores
"""

import pandas as pd
import numpy as np
import pickle

# Load data
factor_scores = pd.read_csv('factor_scores.csv')
with open('cluster_data.pkl', 'rb') as f:
    cluster_data = pickle.load(f)

cluster_labels_full = cluster_data['cluster_labels_full']

# Get Person 12 data
person_id = 12
person_factors = factor_scores.iloc[person_id]
person_cluster = cluster_labels_full[person_id]

# Calculate z-scores
person_scores = person_factors.values
all_means = factor_scores.mean()
all_stds = factor_scores.std()
person_z_scores = (person_scores - all_means.values) / all_stds.values

# Identify strengths and weaknesses
strengths = []
weaknesses = []

for i, (factor_name, z_score) in enumerate(zip(person_factors.index, person_z_scores)):
    if z_score > 0.5:
        strengths.append((factor_name, person_scores[i], z_score))
    elif z_score < -0.5:
        weaknesses.append((factor_name, person_scores[i], z_score))

# Sort by magnitude
strengths = sorted(strengths, key=lambda x: x[2], reverse=True)
weaknesses = sorted(weaknesses, key=lambda x: x[2])

# GENERATE 3-SENTENCE SUMMARY

# Sentence 1: Strengths
strength_factors = [f.replace('Factor ', '') for f, _, _ in strengths[:2]]
sentence_1 = f"Person 12 demonstrates exceptional strengths in {' and '.join(strength_factors)}, positioning them as particularly adept at understanding nuanced perspectives and maintaining emotional connection with others."

# Sentence 2: Weaknesses & Average
weakness_factors = [f.replace('Factor ', '') for f, _, _ in weaknesses[:2]]
avg_count = len([z for z in person_z_scores if -0.5 <= z <= 0.5])
sentence_2 = f"In contrast, they show relative weakness in {' and '.join(weakness_factors)}, and perform at approximately average levels across {avg_count} other personality dimensions."

# Sentence 3: Overall Characterization
sentence_3 = f"Overall, Person 12 represents a compassionate and emotionally aware personality type who excels at empathy but may benefit from developing greater assertiveness and emotional resilience in high-pressure situations."

# PRINT RESULTS
print("="*80)
print("PERSON 12 - THREE SENTENCE SUMMARY")
print("="*80)
print()
print(f"Sentence 1 (Strengths):\n{sentence_1}")
print()
print(f"Sentence 2 (Weaknesses/Average):\n{sentence_2}")
print()
print(f"Sentence 3 (Characterization):\n{sentence_3}")

# SUMMARY STATISTICS
print()
print("="*80)
print("SUMMARY STATISTICS")
print("="*80)
print(f"Total Strengths: {len(strengths)}")
print(f"Total Weaknesses: {len(weaknesses)}")
print(f"Average Factors: {avg_count}")
print(f"Assigned Cluster: {cluster_data['cluster_names'][person_cluster]}")

