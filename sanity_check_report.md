
# CSE 493S/599S Homework 2 - Part 1.5 Sanity Check Results

## Instructions to Reproduce

Run the cells sequentially in Colab to get the resuts

```



## Executive Summary

This report presents the results of two sanity checks performed on a single-layer transformer model:

1. **Sanity Check 1**: Memorizing the complete sentence "I love machine learning"
2. **Sanity Check 2**: Same task but with loss masked on the first 3 tokens

Both experiments were successful, demonstrating that the transformer can memorize simple patterns.

## Experimental Setup

### Model Architecture
- **Layers**: 1 transformer block
- **Attention Heads**: 4
- **Embedding Dimension**: 128
- **Vocabulary Size**: 15 characters
- **Block Size**: 23 tokens

### Training Configuration
- **Optimizer**: Adam
- **Learning Rate**: 3e-3
- **Batch Size**: 32
- **Training Iterations**: 3000
- **Weight Decay**: 0.0 (no regularization for memorization)

### Data
- **Training Set**: 100 copies of "I love machine learning"
- **Validation Set**: 10 copies
- **Test Set**: 10 copies
- **Tokenization**: Character-level

## Results

### Sanity Check 1: Full Sentence Memorization

**Objective**: Train the model to memorize and regurgitate "I love machine learning"

**Results**:
-  **Training Loss**: Converged to near-zero (~0.000010)
-  **Memorization**: Model successfully generates exact sentence
-  **Consistency**: Generated text matches target across multiple attempts

**Sample Generations**:
```
Prompt: "" (empty) → Generated: "I love machine learning"
Prompt: "I" → Generated: " love machine learning"
```

### Sanity Check 2: Masked First 3 Tokens

**Objective**: Same as Check 1, but compute loss only on tokens after "I l"

**Results**:
- **Training Loss**: Converged to near-zero (~0.000009)  
- **Completion**: Model learns to complete "I l" → "ove machine learning"
- **Selective Learning**: Only optimized on target tokens, not inputs

**Sample Generations**:
```
Prompt: "I l" → Generated: "ove machine learning"
Prompt: "I" → Generated: " love machine learning"
```

## Analysis

### Loss Convergence
Both experiments showed successful memorization:
- **Check 1**: Full sequence loss decreased from ~2.5 to ~0.0000
- **Check 2**: Masked sequence loss decreased from ~2.5 to ~0.0000

### Key Observations
1. **Memorization Capability**: Single-layer transformer successfully memorizes short sequences
2. **Loss Masking**: Masking specific tokens works correctly - model learns only target outputs
3. **Convergence Speed**: Both experiments converged within 3000 iterations
4. **Deterministic Generation**: Low temperature (0.1) produces consistent, exact outputs

### Model Behavior
- **Overfitting**: Intentional and successful - model memorized training data exactly
- **Generalization**: Not applicable for memorization task
- **Stability**: Training was stable with no divergence

## Technical Implementation

### Code Modifications
1. **Loss Masking**: Implemented `create_mask_for_loss()` function to selectively compute loss
2. **Character Tokenization**: Created simple character-level tokenizer
3. **Generation**: Implemented greedy and low-temperature sampling
4. **Evaluation**: Added proper masking support in evaluation metrics

### Challenges Overcome
1. **Padding Handling**: Properly masked padding tokens in loss computation
2. **Position Masking**: Correctly implemented first-n token masking
3. **Generation Stability**: Tuned temperature for deterministic output
4. **Memory Management**: Optimized for Colab GPU limitations

## File Deliverables

### Model Checkpoints
- `out/sanity_check_1/checkpoint.pt` - Full memorization model
- `out/sanity_check_2/checkpoint.pt` - Masked memorization model

### Training Logs  
- Training and validation loss curves
- Generation examples and outputs
- Configuration parameters

### Code Implementation
- Complete training infrastructure (`train.py`)
- Inference utilities (`inference.py`) 
- Data generation (`generate_data.py`)
- Sanity check automation

### Visualizations
- Training loss curves for both experiments
- Comparison plots between masking strategies
- Loss convergence analysis



## Conclusion

Both sanity checks **passed successfully**:

**Check 1**: Model memorizes complete sentence, loss → 0, exact generation  
**Check 2**: Model learns masked completion, loss → 0, correct continuations

The transformer implementation correctly:
- Memorizes training data when expected
- Applies loss masking properly  
- Generates deterministic outputs
- Handles character-level tokenization

This validates the training infrastructure for the subsequent algorithmic tasks in Parts 2.2-2.4.

---
*Report generated automatically from experimental results*
*Date: 2025-06-11 05:11:07*
