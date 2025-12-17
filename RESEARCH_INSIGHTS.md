# Research Insights from Stage 1 Bias Analysis Log

## Executive Summary
**Execution Time**: 2 minutes 32 seconds (152.71 seconds)  
**Dataset**: Adult Income (24,128 training, 6,032 test samples)  
**Gender Distribution**: Male: 16,355 (67.8%), Female: 7,773 (32.2%) - **Imbalanced dataset**

---

## 1. Model Training Performance

### Training Loss Progression (300 epochs)
- **Epoch 0**: 0.401248
- **Epoch 20**: 0.283353 (-29.4% reduction)
- **Epoch 40**: 0.243671 (-39.3% reduction)
- **Epoch 60**: 0.214322 (-46.6% reduction)
- **Epoch 80**: 0.179138 (-55.4% reduction)
- **Epoch 100**: 0.168385 (-58.1% reduction)
- **Epoch 120**: 0.158648 (-60.5% reduction)
- **Epoch 140**: 0.147044 (-63.4% reduction)
- **Epoch 160**: 0.150679 (slight increase - possible overfitting)
- **Epoch 180**: 0.146699 (-63.5% reduction)
- **Epoch 200**: 0.136828 (-65.9% reduction)
- **Epoch 220**: 0.126682 (-68.5% reduction)
- **Epoch 240**: 0.139318 (increase - overfitting signal)
- **Epoch 260**: 0.138920
- **Epoch 280**: 0.120657 (-70.0% reduction) - **Final loss**

**Observation**: Model shows signs of overfitting after epoch 140 (loss increases at epoch 160, 240).

---

## 2. Bias Detection Results

### Gender Classification from Activations
- **Training Accuracy**: 0.7871 (78.71%)
- **Test Accuracy**: 0.7822 (78.22%)
- **Baseline (Random)**: 0.5000 (50%)
- **Bias Detected**: **YES** (threshold: 0.55)

**Per-Class Performance**:
- **Male Accuracy**: 0.8178 (81.78%)
- **Female Accuracy**: 0.7226 (72.26%)
- **Gap**: 9.52 percentage points (model better at predicting males)

**Critical Finding**: The model's activations leak gender information with **78% accuracy**, indicating strong gender bias encoded in the penultimate layer.

---

## 3. Top Biased Neurons (Importance Ranking)

### Top 10 Most Biased Neurons
1. **Neuron 34**: Importance = 4.128 (highest) - Direction: Male
2. **Neuron 56**: Importance = 3.468 - Direction: Male
3. **Neuron 3**: Importance = 3.008 - Direction: Female
4. **Neuron 1**: Importance = 2.913 - Direction: Male
5. **Neuron 31**: Importance = 2.628 - Direction: Female
6. **Neuron 17**: Importance = 2.575 - Direction: Female
7. **Neuron 26**: Importance = 2.475 - Direction: Female
8. **Neuron 50**: Importance = 2.459 - Direction: Male
9. **Neuron 61**: Importance = 2.433 - Direction: Male
10. **Neuron 14**: Importance = 2.230 - Direction: Male

**Key Observations**:
- **Mixed Directions**: 6 neurons favor Male, 4 favor Female
- **Neuron 34** is the most critical (4.13x mean importance)
- **Neuron 3** is the most Female-favoring (3.01x mean importance)

### Neuron Importance Distribution
- **Highest**: 4.128 (Neuron 34)
- **Lowest (non-zero)**: 0.123 (Neuron 48)
- **Zero importance**: Neurons 13, 19, 22, 29, 35, 36, 39, 40, 44, 55, 59, 63 (12 neurons have zero importance)

---

## 4. Accuracy vs. Discrimination Trade-off Analysis

### Baseline Performance
- **Income Classification Accuracy**: 0.8286 (82.86%)
- **Counterfactual Discrimination**: 819/6032 = **13.58%**
- **Prediction Confidence**:
  - Probabilities > 0.5: 1,389 instances, Avg = 0.8706
  - Probabilities < 0.5: 4,643 instances, Avg = 0.0504
  - **Clear separation**: Model is confident in predictions

### Masking Results Summary

| Neurons Masked | Accuracy | Discrimination | Change in Disc. | Avg Prob > 0.5 | Avg Prob < 0.5 | Instances > 0.5 |
|----------------|----------|----------------|-----------------|----------------|----------------|-----------------|
| **Baseline** | 0.8286 | 13.58% | - | 0.8706 | 0.0504 | 1,389 |
| **10** | 0.8143 | 14.52% | **+0.94%** ⚠️ | 0.8738 | 0.0704 | 1,677 |
| **20** | 0.7886 | 15.09% | **+1.51%** ⚠️ | 0.8648 | 0.0980 | 2,036 |
| **30** | 0.7994 | 15.00% | **+1.42%** ⚠️ | 0.8269 | 0.1203 | 1,871 |
| **40** | 0.8205 | 14.14% | +0.56% | 0.7454 | 0.1397 | 1,540 |
| **45** | 0.8094 | 14.46% | +0.88% | 0.7145 | 0.1959 | 1,705 |
| **50** | 0.8249 | **11.42%** | **-2.16%** ✅ | 0.6247 | 0.2452 | 1,025 |
| **54** | 0.7512 | **0.00%** | **-13.58%** ✅ | N/A (all < 0.5) | 0.4789 | 0 |
| **58** | 0.7512 | 0.00% | -13.58% | N/A | 0.4789 | 0 |
| **60** | 0.7512 | 0.00% | -13.58% | N/A | 0.4789 | 0 |
| **62** | 0.7512 | 0.00% | -13.58% | N/A | 0.4789 | 0 |

---

## 5. Critical Findings

### 5.1 The "Inverse Fairness" Paradox
**Finding**: Masking the top 10-30 "most biased" neurons **INCREASES** discrimination instead of reducing it.

- **Top 10 masked**: Discrimination increases from 13.58% → 14.52% (+0.94%)
- **Top 20 masked**: Discrimination increases to 15.09% (+1.51%)
- **Top 30 masked**: Discrimination remains high at 15.00% (+1.42%)

**Hypothesis**: Removing high-importance neurons reduces model confidence, making predictions more susceptible to remaining subtle gender signals.

### 5.2 The "Sweet Spot" at k=50
**Finding**: Masking 50 neurons achieves the **best fairness-accuracy trade-off**.

- **Accuracy**: 0.8249 (only -0.37% from baseline)
- **Discrimination**: 11.42% (**-2.16% reduction** from baseline)
- **Confidence**: Average probability for > 0.5 drops to 0.6247 (from 0.8706)
- **Still functional**: 1,025 instances predicted > 0.5

**This is the optimal masking level for practical deployment.**

### 5.3 The "Degenerate State" Threshold
**Finding**: At k=54, the model enters a degenerate state.

**Before k=54**:
- Model makes diverse predictions
- Confidence varies across instances
- Discrimination exists but is measurable

**At k=54+**:
- **All predictions converge to 0.4789** (exactly the same value!)
- **Zero discrimination** (but model is useless)
- **Accuracy drops to 0.7512** (matches majority class baseline ~75%)
- Model effectively predicts "Income <= 50k" for everyone

**Critical Threshold**: Between k=50 and k=54, the model collapses.

---

## 6. Confidence Degradation Pattern

### Average Probability Trends

**For Predictions > 0.5**:
- Baseline: 0.8706 (high confidence)
- k=10: 0.8738 (slight increase - more instances cross threshold)
- k=20: 0.8648 (decreasing)
- k=30: 0.8269 (significant drop)
- k=40: 0.7454 (major drop)
- k=45: 0.7145 (approaching threshold)
- k=50: 0.6247 (low confidence, near threshold)
- k=54+: **No predictions > 0.5** (complete collapse)

**For Predictions < 0.5**:
- Baseline: 0.0504 (very low, confident negatives)
- k=10: 0.0704 (increasing)
- k=20: 0.0980 (doubled)
- k=30: 0.1203 (2.4x baseline)
- k=40: 0.1397 (2.8x baseline)
- k=45: 0.1959 (3.9x baseline)
- k=50: 0.2452 (4.9x baseline)
- k=54+: 0.4789 (9.5x baseline - **all predictions converge here**)

**Pattern**: As more neurons are masked, the model becomes less confident in both positive and negative predictions, eventually collapsing to a constant output.

---

## 7. Instance Count Analysis

### Number of Instances Predicted > 0.5

| Masking Level | Instances > 0.5 | Percentage | Trend |
|---------------|-----------------|-----------|-------|
| Baseline | 1,389 | 23.0% | - |
| k=10 | 1,677 | 27.8% | **+288** (more positives) |
| k=20 | 2,036 | 33.8% | **+647** (significant increase) |
| k=30 | 1,871 | 31.0% | -165 (decrease) |
| k=40 | 1,540 | 25.5% | -331 |
| k=45 | 1,705 | 28.3% | +165 |
| k=50 | 1,025 | 17.0% | -680 (major drop) |
| k=54+ | 0 | 0.0% | **Complete collapse** |

**Observation**: Initially (k=10-20), masking increases positive predictions, suggesting the model compensates by being more lenient. After k=30, positive predictions decrease, and at k=54, they disappear entirely.

---

## 8. Research Implications

### 8.1 Neuron Masking is NOT a Simple Solution
**Finding**: Simply removing "biased" neurons does not guarantee fairness improvement.

- **Low masking (k=10-30)**: Actually **worsens** discrimination
- **Medium masking (k=40-50)**: Can improve fairness if done carefully
- **High masking (k=54+)**: Eliminates discrimination but destroys model utility

### 8.2 The Importance of Direction
**Finding**: The current importance metric ignores the **direction** of bias.

- Some neurons favor Male (positive direction)
- Some neurons favor Female (negative direction)
- Removing both types simultaneously may cancel out beneficial effects

**Recommendation**: Consider masking only neurons with the same direction, or use directional importance scores.

### 8.3 Confidence as a Mediator
**Finding**: Model confidence is a critical mediator between masking and discrimination.

- **High confidence** (baseline): Clear predictions, moderate discrimination
- **Medium confidence** (k=40-50): Fuzzy predictions, discrimination varies
- **Low confidence** (k=54+): Constant predictions, zero discrimination but useless

**The "sweet spot" (k=50) balances confidence and fairness.**

### 8.4 The Degenerate State
**Finding**: There's a critical threshold where the model collapses.

- **Between k=50 and k=54**: Model transitions from functional to degenerate
- **At k=54+**: All activations become zero (except 2-4 neurons)
- **Result**: Constant output (0.4789) for all inputs

**This suggests the model has ~4-6 "critical" neurons that maintain functionality.**

---

## 9. Statistical Summary

### Accuracy Distribution
- **Highest**: 0.8286 (baseline)
- **Lowest (functional)**: 0.7512 (k=54+)
- **Best trade-off**: 0.8249 (k=50) - only 0.37% drop

### Discrimination Distribution
- **Highest**: 15.09% (k=20)
- **Lowest**: 0.00% (k=54+)
- **Best improvement**: 11.42% (k=50) - 2.16% reduction from baseline

### Confidence Spread
- **Baseline**: 0.8706 (high) vs 0.0504 (low) - **Clear separation**
- **k=50**: 0.6247 (medium) vs 0.2452 (medium) - **Compressed range**
- **k=54+**: N/A vs 0.4789 (constant) - **No separation**

---

## 10. Recommendations for Future Research

### 10.1 Direction-Aware Masking
Instead of masking by magnitude alone, consider:
- Mask only Male-favoring neurons
- Mask only Female-favoring neurons
- Compare results to understand directional effects

### 10.2 Gradual Masking
Instead of binary masking (0.0 or 1.0), try:
- Partial masking (0.25, 0.5, 0.75)
- May preserve some functionality while reducing bias

### 10.3 Retraining After Masking
Current approach: Mask at inference time
Alternative: Retrain model with masked neurons
- May allow model to adapt and compensate
- Could achieve better fairness-accuracy trade-off

### 10.4 Neuron Repair Instead of Removal
Instead of zeroing neurons, consider:
- Adjusting neuron weights to reduce gender correlation
- Fine-tuning specific neurons
- May preserve functionality better

### 10.5 Multi-Objective Optimization
Consider optimizing for:
- Accuracy
- Fairness (discrimination)
- Confidence calibration
- Simultaneously, not sequentially

---

## 11. Key Numbers for Papers/Tables

### Dataset Statistics
- Training samples: 24,128
- Test samples: 6,032
- Gender imbalance: 67.8% Male, 32.2% Female
- Input features: 102 (after one-hot encoding)

### Model Architecture
- Layers: Input → 256 → 128 → 64 → 1
- Total parameters: ~(102×256 + 256×128 + 128×64 + 64×1) ≈ 60,000
- Training epochs: 300
- Final loss: 0.120657

### Bias Detection
- Gender classification accuracy: 78.22%
- Bias detection threshold: 55%
- **Bias confirmed**: Yes (28.22% above random)

### Optimal Masking
- **Best k**: 50 neurons
- **Accuracy**: 82.49%
- **Discrimination**: 11.42%
- **Improvement**: -2.16% discrimination, -0.37% accuracy

### Degenerate Threshold
- **Critical k**: 54 neurons
- **Collapse point**: Between k=50 and k=54
- **Degenerate accuracy**: 75.12% (matches majority class)
- **Degenerate output**: Constant 0.4789 for all inputs

---

## 12. Surprising Discoveries

1. **Inverse Fairness Effect**: Removing biased neurons initially **increases** discrimination
2. **Non-Monotonic Behavior**: Discrimination doesn't decrease monotonically with masking
3. **Confidence Collapse**: Model loses confidence before losing accuracy
4. **Optimal Point Exists**: k=50 provides best trade-off, not k=0 or k=60
5. **Degenerate State**: Model collapses to constant output at k=54, not gradually
6. **Gender Classification Gap**: 9.52% accuracy gap between Male and Female prediction

---

## 13. Conclusion

The log reveals that **neuron masking is a complex intervention** with non-linear effects:

- **Low masking** (k=10-30): Counterproductive - increases discrimination
- **Medium masking** (k=40-50): Potentially beneficial - can reduce discrimination
- **High masking** (k=54+): Destructive - eliminates discrimination but destroys utility

**The key insight**: Simply removing "biased" neurons is not sufficient. The **direction** of bias, **model confidence**, and **neuron interactions** all play critical roles in determining the outcome.

**Practical recommendation**: For the Adult Income dataset, masking **50 neurons** provides the best balance between fairness and accuracy, achieving a 2.16% reduction in discrimination with only a 0.37% accuracy drop.
