
# Part 2.2: Warmup Experiments Results Report

## Executive Summary

Successfully completed 24 warmup experiments on addition and subtraction tasks.

**Key Results:**
- Average Test Accuracy: 0.9562
- Best Test Accuracy: 0.9867
- Total Training Time: 1255.8 seconds (0.3 hours)

**Best Performing Configuration:**
- Task: Addition
- Prime (p): 97
- Layers: 2
- Seed: 3
- Test Accuracy: 0.9867

## Experimental Setup

**Model Architecture:**
- Dimensions: 128 embedding, 4 attention heads, 512 FFN dimension
- Layers: 1 and 2 transformer blocks
- Parameters: ~0.6M (1-layer), ~1.2M (2-layer)

**Training Configuration:**
- Optimizer: AdamW with cosine learning rate schedule
- Learning Rate: 3e-4
- Batch Size: 64
- Training Steps: Up to 100k (reduced to 20k for demo)
- Loss: Cross-entropy on answer tokens only

**Tasks and Data:**
- Addition and Subtraction modular arithmetic
- Prime values: p = 97, 113
- Data splits: 50% train, 25% val, 25% test
- Evaluation: Exact match accuracy on answers

## Detailed Results

### Results by Configuration

| Task | p | Layers | Seeds | Avg Test Acc | Best Test Acc | Std Dev |
|------|---|--------|-------|--------------|---------------|---------|
| Addition | 97 | 1 | 3 | 0.9774 | 0.9801 | 0.0022 |
| Addition | 97 | 2 | 3 | 0.9859 | 0.9867 | 0.0007 |
| Addition | 113 | 1 | 3 | 0.9425 | 0.9442 | 0.0012 |
| Addition | 113 | 2 | 3 | 0.9616 | 0.9628 | 0.0016 |
| Subtraction | 97 | 1 | 3 | 0.9555 | 0.9586 | 0.0024 |
| Subtraction | 97 | 2 | 3 | 0.9737 | 0.9743 | 0.0009 |
| Subtraction | 113 | 1 | 3 | 0.8838 | 0.8967 | 0.0092 |
| Subtraction | 113 | 2 | 3 | 0.9695 | 0.9722 | 0.0037 |


### Key Findings

**Task Performance:**
- Addition: 0.9669 average accuracy
- Subtraction: 0.9456 average accuracy

**Architecture Impact:**
- 1-layer models: 0.9398 average accuracy
- 2-layer models: 0.9727 average accuracy
- 2-layer improvement: +3.5% over 1-layer


**Training Efficiency:**
- Average training time: 52.3 seconds per experiment
- Time to convergence varies significantly based on random initialization

**Reproducibility:**
- Multiple seeds show consistent performance
- Best practices: Use multiple random restarts for reliable results

## Conclusions

1. **Feasibility**: Transformer models can successfully learn modular arithmetic
2. **Architecture**: 2-layer models consistently outperform 1-layer models  
3. **Task Difficulty**: Addition and subtraction show similar learning patterns
4. **Scale Effects**: Both p=97 and p=113 are learnable with sufficient training

## Next Steps

 **Ready for Part 2.3**: Grokking experiments on division task
- Division expected to be significantly more challenging
- May require longer training to observe grokking phenomenon

## Files Generated

- Model checkpoints: `out/warmup_*/best_model.pt`
- Training curves: `reports/warmup_experiments_analysis.png`
- This report: `reports/warmup_experiments_report.md`

---
*Generated from 24 experimental runs*
*Report timestamp: 2025-06-11 03:30:05*
