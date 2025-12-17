# Detailed Analysis Results

## 1. OUTGOING WEIGHTS DISTRIBUTION (Second-to-Last to Last Layer)

| Weight Range | Number of Neurons |
|--------------|-------------------|
| 0.00000 - 0.08665 | 11 neurons |
| 0.08665 - 0.17331 | 0 neurons |
| 0.17331 - 0.25996 | 0 neurons |
| 0.25996 - 0.34661 | 0 neurons |
| 0.34661 - 0.43326 | 0 neurons |
| 0.43326 - 0.51992 | 0 neurons |
| 0.51992 - 0.60657 | 0 neurons |
| 0.60657 - 0.69322 | 0 neurons |
| 0.69322 - 0.77988 | 3 neurons |
| 0.77988 - 0.86653 | 11 neurons |
| 0.86653 - 0.95318 | 11 neurons |
| 0.95318 - 1.03984 | 10 neurons |
| 1.03984 - 1.12649 | 7 neurons |
| 1.12649 - 1.21314 | 3 neurons |
| 1.21314 - 1.29979 | 2 neurons |
| 1.29979 - 1.38645 | 1 neurons |
| 1.38645 - 1.47310 | 0 neurons |
| 1.47310 - 1.55975 | 0 neurons |
| 1.55975 - 1.64641 | 1 neurons |
| 1.64641 - 1.73306 | 4 neurons |

**Summary**: Most weights are concentrated in the 0.78-1.73 range (54 neurons), with 11 neurons having very small weights (<0.087).

---

## 2. SVM COEFFICIENTS DISTRIBUTION (with Direction)

### Female-Favoring (Negative Coefficients)
| Coefficient Range | Number of Neurons |
|-------------------|-------------------|
| -13.27990 - -12.02567 | 1 neurons |
| -10.77143 - -9.51719 | 4 neurons |
| -8.26295 - -7.00871 | 2 neurons |
| -7.00871 - -5.75447 | 4 neurons |
| -5.75447 - -4.50023 | 6 neurons |
| -4.50023 - -3.24600 | 2 neurons |
| -3.24600 - -1.99176 | 5 neurons |
| -1.99176 - -0.73752 | 5 neurons |

**Total Female-favoring neurons: 29**

### Zero Coefficients
**9 neurons** (no gender signal)

### Male-Favoring (Positive Coefficients)
| Coefficient Range | Number of Neurons |
|-------------------|-------------------|
| 0.27562 - 2.31771 | 7 neurons |
| 2.31771 - 4.35981 | 7 neurons |
| 4.35981 - 6.40191 | 2 neurons |
| 6.40191 - 8.44401 | 3 neurons |
| 8.44401 - 10.48610 | 3 neurons |
| 10.48610 - 12.52820 | 1 neurons |
| 12.52820 - 14.57030 | 1 neurons |
| 14.57030 - 16.61240 | 1 neurons |
| 18.65449 - 20.69659 | 1 neurons |

**Total Male-favoring neurons: 26**

**Summary**: 
- **Female-favoring**: 29 neurons (coefficients range from -13.28 to -0.74)
- **Male-favoring**: 26 neurons (coefficients range from 0.28 to 20.70)
- **Zero signal**: 9 neurons
- **Highest coefficient**: 20.70 (Neuron 34, Male-favoring)
- **Lowest coefficient**: -13.28 (Female-favoring)

---

## 3. PROBABILITY DISTRIBUTION FOR EACH K

### k = 10 (Baseline comparison: 1,389 instances > 0.5)
| Probability Range | Instances |
|-------------------|-----------|
| 1.0 - 0.9 | 1,007 instances |
| 0.9 - 0.8 | 199 instances |
| 0.8 - 0.7 | 152 instances |
| 0.7 - 0.6 | 157 instances |
| 0.6 - 0.5 | 162 instances |
| 0.5 - 0.4 | 228 instances |
| 0.4 - 0.3 | 261 instances |
| 0.3 - 0.2 | 235 instances |
| 0.2 - 0.1 | 194 instances |
| 0.1 - 0.0 | 3,437 instances |

**Observation**: High confidence predictions (1.0-0.9) dominate with 1,007 instances. Low confidence (0.1-0.0) has 3,437 instances.

---

### k = 20
| Probability Range | Instances |
|-------------------|-----------|
| 1.0 - 0.9 | 1,199 instances |
| 0.9 - 0.8 | 222 instances |
| 0.8 - 0.7 | 181 instances |
| 0.7 - 0.6 | 202 instances |
| 0.6 - 0.5 | 232 instances |
| 0.5 - 0.4 | 362 instances |
| 0.4 - 0.3 | 285 instances |
| 0.3 - 0.2 | 204 instances |
| 0.2 - 0.1 | 305 instances |
| 0.1 - 0.0 | 2,840 instances |

**Observation**: High confidence (1.0-0.9) **increased** to 1,199. Low confidence (0.1-0.0) **decreased** to 2,840.

---

### k = 30
| Probability Range | Instances |
|-------------------|-----------|
| 1.0 - 0.9 | 874 instances |
| 0.9 - 0.8 | 253 instances |
| 0.8 - 0.7 | 232 instances |
| 0.7 - 0.6 | 238 instances |
| 0.6 - 0.5 | 274 instances |
| 0.5 - 0.4 | 516 instances |
| 0.4 - 0.3 | 295 instances |
| 0.3 - 0.2 | 266 instances |
| 0.2 - 0.1 | 380 instances |
| 0.1 - 0.0 | 2,704 instances |

**Observation**: High confidence (1.0-0.9) **decreased** to 874. More instances in middle ranges (0.5-0.4: 516).

---

### k = 40
| Probability Range | Instances |
|-------------------|-----------|
| 1.0 - 0.9 | 424 instances |
| 0.9 - 0.8 | 170 instances |
| 0.8 - 0.7 | 221 instances |
| 0.7 - 0.6 | 299 instances |
| 0.6 - 0.5 | 426 instances |
| 0.5 - 0.4 | 679 instances |
| 0.4 - 0.3 | 315 instances |
| 0.3 - 0.2 | 336 instances |
| 0.2 - 0.1 | 489 instances |
| 0.1 - 0.0 | 2,673 instances |

**Observation**: High confidence (1.0-0.9) **halved** to 424. Peak shifts to 0.5-0.4 range (679 instances).

---

### k = 45
| Probability Range | Instances |
|-------------------|-----------|
| 1.0 - 0.9 | 350 instances |
| 0.9 - 0.8 | 180 instances |
| 0.8 - 0.7 | 249 instances |
| 0.7 - 0.6 | 334 instances |
| 0.6 - 0.5 | 592 instances |
| 0.5 - 0.4 | 858 instances |
| 0.4 - 0.3 | 480 instances |
| 0.3 - 0.2 | 534 instances |
| 0.2 - 0.1 | 679 instances |
| 0.1 - 0.0 | 1,776 instances |

**Observation**: High confidence (1.0-0.9) drops to 350. Distribution becomes more uniform, with peak at 0.5-0.4 (858 instances).

---

### k = 50
| Probability Range | Instances |
|-------------------|-----------|
| 1.0 - 0.9 | 74 instances |
| 0.9 - 0.8 | 80 instances |
| 0.8 - 0.7 | 86 instances |
| 0.7 - 0.6 | 166 instances |
| 0.6 - 0.5 | 619 instances |
| 0.5 - 0.4 | 1,318 instances |
| 0.4 - 0.3 | 723 instances |
| 0.3 - 0.2 | 774 instances |
| 0.2 - 0.1 | 788 instances |
| 0.1 - 0.0 | 1,404 instances |

**Observation**: High confidence (1.0-0.9) **collapses** to only 74 instances. Distribution heavily skewed toward middle ranges (0.5-0.4: 1,318 instances).

---

### k = 54, 58, 60 (Degenerate State)
| Probability Range | Instances |
|-------------------|-----------|
| 1.0 - 0.9 | 0 instances |
| 0.9 - 0.8 | 0 instances |
| 0.8 - 0.7 | 0 instances |
| 0.7 - 0.6 | 0 instances |
| 0.6 - 0.5 | 0 instances |
| 0.5 - 0.4 | 6,032 instances (ALL) |
| 0.4 - 0.3 | 0 instances |
| 0.3 - 0.2 | 0 instances |
| 0.2 - 0.1 | 0 instances |
| 0.1 - 0.0 | 0 instances |

**Observation**: **Complete collapse** - all 6,032 instances have probability 0.4789 (within 0.5-0.4 range). Model outputs constant value.

---

## 4. MALE vs FEMALE FAVORING NEURONS DROPPED FOR EACH K

**Total Counts**:
- Male-favoring neurons: 26
- Female-favoring neurons: 29
- Zero-direction neurons: 5

### k = 10
- **Male-favoring dropped**: 6 (23.08% or 0.2308 ratio of total 26)
- **Female-favoring dropped**: 4 (13.79% or 0.1379 ratio of total 29)
- **Zero-direction dropped**: 0

**Observation**: More Male-favoring neurons removed initially (23% vs 14%).

---

### k = 20
- **Male-favoring dropped**: 8 (30.77% or 0.3077 ratio of total 26)
- **Female-favoring dropped**: 12 (41.38% or 0.4138 ratio of total 29)
- **Zero-direction dropped**: 0

**Observation**: Now more Female-favoring neurons removed (41% vs 31%). This explains why positives increased at k=20.

---

### k = 30
- **Male-favoring dropped**: 12 (46.15% or 0.4615 ratio of total 26)
- **Female-favoring dropped**: 18 (62.07% or 0.6207 ratio of total 29)
- **Zero-direction dropped**: 0

**Observation**: Female-favoring neurons being removed faster (62% vs 46%).

---

### k = 40
- **Male-favoring dropped**: 18 (69.23% or 0.6923 ratio of total 26)
- **Female-favoring dropped**: 22 (75.86% or 0.7586 ratio of total 29)
- **Zero-direction dropped**: 0

**Observation**: Both groups heavily depleted, but Female-favoring more so (76% vs 69%).

---

### k = 45
- **Male-favoring dropped**: 20 (76.92% or 0.7692 ratio of total 26)
- **Female-favoring dropped**: 25 (86.21% or 0.8621 ratio of total 29)
- **Zero-direction dropped**: 0

**Observation**: 86% of Female-favoring neurons removed, only 77% of Male-favoring.

---

### k = 50
- **Male-favoring dropped**: 22 (84.62% or 0.8462 ratio of total 26)
- **Female-favoring dropped**: 28 (96.55% or 0.9655 ratio of total 29)
- **Zero-direction dropped**: 0

**Observation**: Almost all Female-favoring neurons gone (97%), but 15% of Male-favoring remain.

---

### k = 54
- **Male-favoring dropped**: 26 (100.00% or 1.0000 ratio of total 26)
- **Female-favoring dropped**: 28 (96.55% or 0.9655 ratio of total 29)
- **Zero-direction dropped**: 0

**Observation**: **ALL Male-favoring neurons removed**. Only 1 Female-favoring neuron remains.

---

### k = 58
- **Male-favoring dropped**: 26 (100.00% or 1.0000 ratio of total 26)
- **Female-favoring dropped**: 29 (100.00% or 1.0000 ratio of total 29)
- **Zero-direction dropped**: 3

**Observation**: **ALL gender-favoring neurons removed**. Only zero-direction neurons remain.

---

### k = 60
- **Male-favoring dropped**: 26 (100.00% or 1.0000 ratio of total 26)
- **Female-favoring dropped**: 29 (100.00% or 1.0000 ratio of total 29)
- **Zero-direction dropped**: 5

**Observation**: **ALL neurons with any gender signal removed**. Model collapses.

---

## Key Insights

### 1. Asymmetric Removal Pattern
- **Early stages (k=10-20)**: More Male-favoring neurons removed initially
- **Middle stages (k=20-40)**: Female-favoring neurons removed faster
- **Late stages (k=50+)**: Female-favoring neurons exhausted first (97% at k=50)
- **Critical point (k=54)**: ALL Male-favoring neurons removed, model collapses

### 2. Probability Distribution Evolution
- **k=10-20**: High confidence (1.0-0.9) **increases** (unmasking effect)
- **k=30-45**: High confidence **decreases**, distribution becomes more uniform
- **k=50**: High confidence **collapses** (only 74 instances in 1.0-0.9 range)
- **k=54+**: Complete collapse - all predictions at 0.4789

### 3. Extreme Range Behavior
As expected:
- **1.0-0.9 range**: Decreases from 1,007 (k=10) → 74 (k=50) → 0 (k=54+)
- **0.1-0.0 range**: Decreases from 3,437 (k=10) → 1,404 (k=50) → 0 (k=54+)
- **Middle ranges (0.5-0.4)**: Increases dramatically, becoming the dominant range

### 4. The Collapse Mechanism
- At k=54: ALL Male-favoring neurons removed, only 1 Female-favoring remains
- This imbalance (no upward pressure, minimal downward pressure) causes model to output constant value
- The constant value (0.4789) is likely the bias term of the output layer when most activations are zero
