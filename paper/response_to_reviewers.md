# Response to Reviewers: OASIS Paper

## Reviewer Comment Q2 Clarification

### Comment Summary
> "OASIS同样需要选择训练drift范围 $q \in \{1,3,5,10,15,20\}$，这何尝不是一种配置？"

### Response

We appreciate the reviewer's concern about the training drift range selection. We clarify that this is a **design-time engineering choice**, not a **runtime configuration burden** on administrators. The distinction is fundamental:

#### 1. Design-Time vs. Runtime Configuration

| Aspect | Prior Methods (STHoles, ISOMER, QuickSel) | OASIS |
|--------|-------------------------------------------|-------|
| **Per-table parameters** | Required at runtime | None |
| **Configuration examples** | Refinement thresholds, entropy weights, mixture counts | N/A |
| **Administrator burden** | Tune per table | Zero |
| **Training drift range** | N/A | Design choice (train once) |

The training drift range $q \in \{1,3,5,10,15,20\}$ is analogous to:
- Choosing a model architecture (e.g., ResNet vs. VGG)
- Selecting a training dataset (e.g., ImageNet vs. CIFAR)
- Setting hyperparameters during model development

**It is NOT analogous to:**
- Per-table parameter tuning
- Runtime convergence thresholds
- Online learning rates

#### 2. "Train Once, Deploy Everywhere"

Once trained, the **same 38K-parameter model** deploys to:
- Any column type (integer, float, date, string)
- Any table schema
- Any database engine (via ONNX/embedded backend)
- Without modification or retraining

This is enabled by the **normalized feature tensor** (§3.2):
- All values normalized to $[0,1]$
- Distribution shape encoded abstractly
- Model-agnostic representation

#### 3. Empirical Validation (Table 3 in revised paper)

We added experiments validating the design choice:

| Training Range | Q-Error at $q=10$ | Worst-case | Generalization |
|---------------|-------------------|------------|----------------|
| Narrow ($q=10$ only) | 1.198 | 3.156 (fails) | Poor |
| Moderate (3 levels) | 1.256 | 1.556 | Moderate |
| **Diverse (6 levels)** | **1.243** | **1.384** | **Excellent** |

**Key Finding:** Training on a diverse range ensures robust generalization to unseen drift intensities ($q=25, 30$). This validates our design choice empirically.

#### 4. Comparison with ML-Based Methods

The reviewer notes that some ML-based methods are not histogram correctors—we fully agree and emphasize **complementarity**:

| Method Type | Output | Integration Path | Relationship to OASIS |
|-------------|--------|------------------|----------------------|
| NeuroCard, DeepDB, MSCN | Point estimates (cardinality) | Replace CBO estimator | **Complementary** |
| Naru | Multi-column selectivity | Custom inference | **Complementary** |
| OASIS | Corrected histograms | Standard CDF interface | **Foundation layer** |

**Synergistic deployment:**
```
OASIS → Corrected Histograms → CBO → Join Cardinality Model → Final Plan
         (single-column)              (multi-column correlation)
```

OASIS operates at the **statistics level**, improving the foundation that all downstream components consume. Learned cardinality estimators can be layered on top for multi-table joins.

### Changes in Revised Paper

1. **Introduction (§1, "Train once, deploy everywhere"):**
   - Added explicit distinction between design-time and runtime configuration
   - Referenced Table 3 validating diverse training

2. **Related Work (§7, "Relationship to Learned Cardinality Estimators"):**
   - New subsection clarifying complementarity
   - Explicit integration path discussion

3. **Experiments (§6.2, "Training drift range"):**
   - New Table 3 showing failure of narrow training
   - Discussion of design choice validation

4. **Future Work (§8):**
   - Added discussion of cascaded deployment with learned estimators

### Conclusion

The training drift range is a **model design parameter** (like architecture choice), not a **deployment configuration** (like per-table tuning). OASIS eliminates all per-table runtime configuration burden through:
- Unified pre-trained model
- Normalized feature representation  
- Zero-shot deployment across columns and schemas

This represents a qualitative shift from prior methods requiring ongoing per-table parameter management.
