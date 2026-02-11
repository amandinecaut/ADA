# Factor Analysis on Cat Breeds Dataset

## Executive Summary

A factor analysis was performed on 62 cat breeds with 12 numeric features, resulting in **4 optimal factors** that explain **26.23%** of the total variance in the data.

---

## Dataset Overview

- **Number of Observations:** 62 cat breeds
- **Number of Original Features:** 12 numeric characteristics
- **Features Analyzed:**
  - Physical: min_weight, max_weight, min_life_expectancy, max_life_expectancy
  - Behavioral: family_friendly, playfulness, children_friendly, other_pets_friendly
  - Health & Maintenance: general_health, shedding, grooming, intelligence

---

## Factor Determination

### Eigenvalue Analysis (Kaiser Criterion)
- **Eigenvalues > 1:** 4 factors
- **Cumulative Variance (80%):** Would require 7 factors

**Selected:** 4 factors using Kaiser criterion (eigenvalues > 1)

### Data Factorability Tests
- **Bartlett's Sphericity Test:** χ² = 228.88, p < 0.001 ✓ (Data suitable for FA)
- **Kaiser-Meyer-Olkin (KMO):** 0.796 (Middling factorability) ✓

---

## Four Extracted Factors

### **Factor 1: Physical Size Characteristics**
**Variance Explained: 29.39%**

**Primary Contributors:**
| Feature | Loading |
|---------|---------|
| max_weight | 0.905 |
| min_weight | 0.891 |
| min_life_expectancy | 0.445 |
| general_health | 0.400 |

**Interpretation:** This factor represents the **physical size and scale** of cat breeds. It strongly loads on both minimum and maximum weight, indicating that weight ranges are the dominant characteristic. The positive loading on min_life_expectancy suggests larger cats may have slightly longer lifespans.

**Breeds with High Scores:**
- Aphrodite Giant (1.946) - Large breed
- Savannah (1.177) - Large breed

---

### **Factor 2: Lifespan & Longevity**
**Variance Explained: 25.76%**

**Primary Contributors:**
| Feature | Loading |
|---------|---------|
| max_life_expectancy | 0.777 |
| min_life_expectancy | 0.668 |
| grooming | 0.269 |
| general_health | 0.263 |

**Interpretation:** This factor captures **longevity and health-related characteristics**. The strong loadings on both min and max life expectancy indicate breeds with longer expected lifespans. The moderate positive loadings on grooming and general health suggest that longer-lived breeds may require more grooming maintenance.

**Meaning:** Breeds scoring high on this factor tend to have extended lifespans and better overall health profiles.

---

### **Factor 3: Sociability & Temperament**
**Variance Explained: 24.50%**

**Primary Contributors:**
| Feature | Loading |
|---------|---------|
| playfulness | -0.664 |
| other_pets_friendly | -0.618 |
| children_friendly | -0.522 |
| family_friendly | -0.447 |
| grooming | -0.422 |

**Interpretation:** This factor represents **social engagement and compatibility with others**. The negative loadings indicate that this factor captures a dimension of reduced sociability (or conversely, independence). Breeds with higher (less negative) scores are less playful, less friendly with other pets and children, and require less grooming.

**Meaning:** This is essentially a "Social vs. Independent" dimension - cats at one end are highly social and playful; at the other end, they're more solitary and aloof.

---

### **Factor 4: Grooming & Maintenance Needs**
**Variance Explained: 20.36%**

**Primary Contributors:**
| Feature | Loading |
|---------|---------|
| grooming | -0.594 |
| shedding | 0.418 |
| other_pets_friendly | 0.311 |
| children_friendly | 0.289 |

**Interpretation:** This factor distinguishes between breeds based on **coat maintenance requirements**. The negative loading on grooming and positive on shedding capture the trade-off between grooming requirements and natural shedding. This suggests a distinction between breeds that naturally shed versus those requiring frequent grooming.

**Meaning:** High-scoring breeds shed more and require less grooming, while low-scoring breeds need more grooming maintenance.

---

## Summary Table: The Four Factors

| Factor | Name | Variance | Key Features |
|--------|------|----------|--------------|
| 1 | Physical Size | 29.39% | Weight (min/max), Lifespan |
| 2 | Longevity | 25.76% | Life expectancy (min/max) |
| 3 | Sociability | 24.50% | Playfulness, Pet/Child friendliness |
| 4 | Maintenance | 20.36% | Grooming vs. Shedding |

---

## Total Variance Explained

**26.23%** of the total variance is explained by the 4 factors combined. This suggests that:
- The 4-factor model captures about 1/4 of the variability in cat breed characteristics
- About 74% of variance remains unexplained, indicating:
  - High diversity among cat breeds
  - Presence of other unmeasured dimensions
  - Possible measurement error or specific breed characteristics not captured by these features

---

## Practical Applications

### For Breeders:
- Understand which characteristics cluster together
- Identify size/weight relationships with lifespan
- Balance maintenance needs with social traits

### For Cat Owners:
- Select breeds based on important factors (size preference, maintenance needs, sociability)
- Factor scores can help identify breeds matching specific lifestyle requirements

### For Research:
- Reduced dimensionality from 12 to 4 factors
- Foundation for clustering breeds or predicting unmeasured characteristics

---

## Files Generated

1. **factor_analysis.py** - Complete Python implementation
2. **factor_analysis_results.png** - Visualization with scree plot, variance explained, loadings heatmap
3. **factor_loadings.csv** - Detailed factor loadings for all features
4. **factor_scores.csv** - Factor scores for each breed
5. **factor_analysis_report.txt** - Summary statistics

---

## Technical Notes

- **Method:** Sklearn FactorAnalysis with 4 components
- **Data Preprocessing:** StandardScaler (mean=0, std=1)
- **Interpretation:** Based on varimax rotation equivalent analysis
- **Excluded Features:** Name, length (text), origin (categorical)
