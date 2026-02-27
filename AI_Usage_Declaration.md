# Declaration of AI Use
## DL Competition 01 - Image Classification

**Authors:** Felipe Castro, Daniel Santana, David Londoño  
**Date:** February 2026  
**Project:** Intel Image Classification using PyTorch

---

## 1. Did you use AI tools?

**Yes**, AI tools were used during the development of this project.

---

## 2. AI Tools Used

### Primary Tool: Claude (Anthropic)
- **Version:** Claude Sonnet 4.5
- **Platform:** claude.ai
- **Access:** Web interface

---

## 3. Tasks for Which AI Was Used  

AI provided explanations of error messages and suggested fixes. We implemented and tested the solutions.

### 3.1 Data Augmentation Guidance 📊

**Description:**  
AI provided advice on data augmentation strategies to improve model generalization.

**Specific Consultations:**
- "What is TrivialAugmentWide and how does it work?"
- "Should I use ColorJitter or TrivialAugmentWide?"
- "What augmentations are appropriate for scene classification?"
- "Why use ImageNet normalization statistics?"

**Recommendations Received from AI:**
1. Use **TrivialAugmentWide** instead of manual augmentation
   - Automatically selects diverse transformations
   - State-of-the-art augmentation strategy
   - Better than manually designed pipelines

2. Add **RandomAdjustSharpness**
   - Enhances edge detection
   - Helps with scene boundary recognition

3. Use **RandomHorizontalFlip** but not vertical
   - Scenes are horizontally symmetric
   - Vertical flip would create unrealistic scenes (upside-down mountains)

**Level of AI Assistance:** Medium (60%)  
AI explained concepts and suggested strategies. We researched further and made final decisions.

### 3.2 Activation Function (GELU) 🧮

**Description:**  
AI explained the GELU activation function and why it might be superior to ReLU.

**Questions Asked:**
- "What is GELU activation function?"
- "Why would GELU be better than ReLU for image classification?"
- "Is GELU computationally expensive?"

**AI Explanations:**
- GELU mathematical definition: `GELU(x) = x × Φ(x)`
- Smoothness properties vs ReLU
- Usage in modern architectures (BERT, GPT, Vision Transformers)
- Empirical benefits in deep networks

**Our Action:**
- Implemented GELU based on AI recommendation
- Compared experimentally: ReLU (67.5%) vs GELU (70.3%)
- Kept GELU due to +2.8% improvement

**Level of AI Assistance:** Medium (50%)  
AI explained the concept, we implemented and validated experimentally.

### 3.3 Learning Rate Scheduler 📈

**Description:**  
AI helped us understand different learning rate schedulers and choose the appropriate one.

**Consultation Topics:**
- "What is the difference between ReduceLROnPlateau and CosineAnnealingLR?"
- "Which scheduler is better for image classification?"
- "How does CosineAnnealingLR work mathematically?"

**AI Guidance:**
- Explained how CosineAnnealingLR creates smooth LR decay
- Provided mathematical formula
- Compared with step-based schedulers
- Recommended CosineAnnealingLR for smooth convergence

**Implementation:**
```python
# Based on AI recommendation:
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=epochs
)
```

**Level of AI Assistance:** Medium (60%)  
AI explained concepts and made recommendations. We implemented and tuned T_max parameter ourselves.

### 3.4 Analysis and Documentation 📝

**Description:**  
AI assisted with analyzing results and creating professional documentation.

**Specific Assistance:**
1. **FInal Results Interpretation**
   - Why validation accuracy > training accuracy (excellent regularization)
   - How to interpret generalization gap
   - Understanding learning curves

2. **Documentation Structure**
   - How to organize technical documentation
   - What details to include in Network Configuration Document
   - How to explain architectural decisions

3. **Writing Clarity**
   - Improving technical explanations
   - Making reproducibility instructions clear
   - Formatting tables and code blocks

**Level of AI Assistance:** High (65%)  
AI helped structure and write documentation. We provided all specific values, results, and architectural decisions.

---

## 4. Which Parts of the Work Are Fully Ours?

### 4.1 Architecture Design ✅

**100% Our Work:**
- Decision to use 3 hidden layers (not 4 or 5)
- Layer dimensions: 4096 → 1024 → 512
- Choice of progressive dropout rates (0.4 → 0.3 → 0.2)
- Number of neurons in each layer

**Reasoning:**
We experimented with different architectures and selected this configuration based on validation performance.

### 4.2 Hyperparameter Selection ✅

**100% Our Work:**
- Batch size: 128 (tested 32, 64, 128)
- Image resolution: 64×64 (tested 32×32, 64×64, 128×128)
- Learning rate: 0.001 (tested 0.0001, 0.001, 0.01)
- Weight decay: 0.01 (tested 0, 0.001, 0.01)
- Number of epochs: 50 (determined by convergence behavior)

**Process:**
We ran multiple experiments and selected values that gave best validation accuracy.

### 4.3 Experimental Iterations ✅

**100% Our Work:**
All experimental runs and comparisons:
- Resolution experiments (32×32 vs 64×64)
- Activation function comparison (ReLU vs GELU)
- Optimizer comparison (Adam vs AdamW)
- Augmentation strategies testing
- Training runs and validation monitoring

**Evidence:**
We have saved checkpoints and training logs from all experiments.

### 4.4 Code Implementation ✅

**100% Our Work:**
- Complete notebook implementation (`Lab_1_final.ipynb`)
- Training and validation functions
- Dataset loading and preprocessing pipelines
- Model instantiation and training loops
- Checkpoint saving mechanism
- Prediction generation for competition test set

**Code Lines Written by Us:**
- All Python code in the notebook (~300 lines)
- Model architecture definition
- Training pipeline
- Data loading configuration

### 4.5 Final Model Selection ✅

**100% Our Work:**
- Monitoring training progress
- Identifying best checkpoint (Epoch 42)
- Analyzing generalization performance
- Deciding when to stop training
- Generating final predictions CSV

### 4.6 Dataset Preparation ✅

**100% Our Work:**
- Downloading dataset from Kaggle
- Organizing folder structure
- Configuring data loaders
- Setting num_workers and batch_size

---

## 5. External Resources Used

### 5.1 Official Documentation

**PyTorch Documentation**
- URL: https://pytorch.org/docs/stable/index.html
- Used for: nn.Module API, optimizer reference, transformation documentation
- Frequency: Multiple times daily

**Torchvision Documentation**
- URL: https://pytorch.org/vision/stable/index.html
- Used for: Data augmentation transforms, ImageFolder usage
- Frequency: Daily

### 5.2 Research Papers

**TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation**
- Authors: Müller & Hutter (2021)
- URL: https://arxiv.org/abs/2103.10158
- Used for: Understanding TrivialAugmentWide implementation

**GELU Activation Function Paper**
- Authors: Hendrycks & Gimpel (2016)
- URL: https://arxiv.org/abs/1606.08415
- Used for: Understanding GELU mathematical properties

**Decoupled Weight Decay Regularization (AdamW)**
- Authors: Loshchilov & Hutter (2019)
- URL: https://arxiv.org/abs/1711.05101
- Used for: Understanding AdamW vs Adam differences

### 5.3 Hardware Platform

**Google Colab**
- URL: https://colab.research.google.com/
- Used for: GPU access (NVIDIA T4)
- Free tier with sufficient resources for our experiments

### 5.4 Community Resources

**PyTorch Forums**
- URL: https://discuss.pytorch.org/
- Used for: torch.compile troubleshooting
- Specific thread: Best practices for DataLoader num_workers

**Stack Overflow**
- URL: https://stackoverflow.com/
- Used for: Specific error resolutions
- Examples: CUDA initialization errors, PIL image loading issues

---

## 6. Breakdown of Work by Source

### Our Original Work (60%)

- Architecture design and selection
- Hyperparameter tuning through experiments
- All code implementation
- Training execution and monitoring
- Results analysis and interpretation
- Model selection decisions
- Experimental comparisons
- Dataset preparation

### AI-Assisted Work (30%)

- Conceptual explanations (GELU, schedulers, augmentation)
- Bug diagnosis and debugging suggestions
- Documentation structure and writing
- Technical writing clarity
- Understanding best practices

### External Resources (10%)

- Official documentation reference
- Research paper background
- Community forum solutions for specific errors
- Dataset source

---


## 7. Honest Assessment of AI Contribution

### What AI Did Well

1. ✅ **Explained Complex Concepts:** GELU, schedulers, augmentation strategies
2. ✅ **Debugging Assistance:** Quickly identified dimension mismatches
3. ✅ **Best Practices:** Recommended modern techniques (AdamW, TrivialAugment)
4. ✅ **Documentation:** Helped structure technical writing clearly

### What AI Did Not Do

1. ❌ **Make Experimental Decisions:** We chose all hyperparameters through testing
2. ❌ **Run Experiments:** We executed all training runs ourselves
3. ❌ **Analyze Results:** We interpreted learning curves and metrics
4. ❌ **Select Final Model:** We decided which checkpoint to use
5. ❌ **Write Code:** We wrote the implementation code ourselves

---

**Signed:**

Felipe Castro Jaimes - 27/02/2026
Daniel Santana - 27/02/2026
David Londoño - 27/02/2026

---

**Document Version:** 1.0  
**Last Updated:** February 2026  
**Status:** Final Declaration

---

*This declaration is submitted in full transparency and academic honesty for DL Competition 01.*
