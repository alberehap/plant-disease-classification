# plant-disease-classification
project

# EfficientNetB3
1. Model Architecture:
    -why choose EfficientNetB3? Because it achieves high accuracy with significantly lower computational cost compared to older models
    
    -EfficientNet, with its compound scaling methodology, had an impact on our understanding of the balance between efficiency and accuracy in deep learning. By intelligently scaling width, depth, and resolution, it offers versatile models adaptable to various hardware constraints.
    
    -source: https://blog.roboflow.com/what-is-efficientnet/
    
2. How To Train The Model:
    -run efficientnetb3_training.ipynb
    -model file google drive link: https://drive.google.com/drive/folders/1qYYAF7P7iAird46PBQJDsL0xplZIx96d

3. Dependencies For Model Training:
    -TensorFlow, Keras, Matplotlib,Numpy.

4. Model Results:
    -Training accuracy: ~95%
    -Validation accuracy: ~92%
    -Test accuracy: ~89%

5. Using Pre-Trained Model
If you want to use the already-trained model:

from tensorflow import keras
model = keras.models.load_model('Models/EfficientNetB3/plant_disease_model.keras')

6. My Role Description:
    -Implemented the ML model using EfficientNet, prepared the training pipeline, trained the model, generated evaluation metrics, and pushed the model files and results to GitHub.
------------------------------------------------------------------------------------
TROUBLESHOOTING: COMMITTING AFTER UPLOADING A LARGE FILE ISSUE FIX!

# Git Troubleshooting: Resolving Large File Issues

## Problem
GitHub push was failing due to large files exceeding the 100MB limit:
- `resnet50_final.h5` (206.85 MB)
- `Models/resnet50_phase1_best.h5` (96.57 MB)
- Various large DLL files from virtual environment

## Solution Steps

### 1. Identify Large Files
```bash
# Find files larger than 50MB
Get-ChildItem -Recurse | Where-Object {$_.Length -gt 50MB} | Format-Table Name, @{Name="Size(MB)";Expression={[math]::Round($_.Length/1MB,2)}} -AutoSize
```

### 2. Remove Large Files from Git History
```bash
# Install git-filter-repo (modern replacement for filter-branch)
pip install git-filter-repo

# Remove all large files from Git history
git filter-repo --path "resnet50_final.h5" --path "libclang.dll" --path "cv2.pyd" --path "_pywrap_tensorflow_common.dll" --path "*.h5" --path "*.keras" --path "*.pyd" --path "*.dll" --invert-paths

# Remove remaining large model files
git filter-repo --path "Models/resnet50_phase1_best.h5" --invert-paths
```

### 3. Reconnect Remote and Push
```bash
# Add remote origin (filter-repo removes remotes)
git remote add origin https://github.com/alberehap/plant-disease-classification.git

# Force push clean repository
git push -u origin main --force
```

## Prevention for Future

### Create Comprehensive .gitignore
```
# Large model files
*.h5
*.keras
*.pkl
*.pth
Models/

# Virtual environment
venv/
env/
*.venv/

# Python cache
__pycache__/
*.pyc
*.pyo
*.pyd

# System files
.DS_Store
Thumbs.db

# Large binaries
*.dll
*.so
*.exe

# Jupyter notebooks
.ipynb_checkpoints/
```

### Best Practices
1. **Never commit virtual environment files** - use `requirements.txt` instead
2. **Use Git LFS for large model files** if they must be versioned
3. **Add large file patterns to .gitignore** before committing
4. **Check file sizes** before pushing with `Get-ChildItem` command

## Key Tools Used
- **git-filter-repo**: Modern tool for rewriting Git history
- **Force push**: Required after history rewriting
- **Comprehensive .gitignore**: Prevents future large file commits

## Result
-Repository successfully pushed to GitHub  
-All large files removed from Git history  
-Clean commit history preserved  
-Proper .gitignore in place to prevent recurrence

This process reduced the repository from ~900MB to ~40MB by removing large model files and virtual environment binaries.


-----------------------------------------------------------------------



# MobileNetV2

1. Model Architecture:

    -Why choose MobileNetV2? Because it is a fast, lightweight, and efficient model designed for mobile and low-power devices while still achieving strong accuracy.

    -MobileNetV2 uses depthwise separable convolutions and inverted residual blocks, making it very efficient for image classification.

    -Source: https://keras.io/api/applications/mobilenet/

2. How To Train The Model:

    -Run mobilenetv2_training.ipynb

    -Model file Google Drive link: https://drive.google.com/drive/folders/1qYYAF7P7iAird46PBQJDsL0xplZIx96d

3. Dependencies For Model Training:

    -TensorFlow, Keras, Matplotlib, NumPy

4. Model Results:

    -Training accuracy: ~86%

    -Validation accuracy: ~88%

    -Test accuracy: ~94%

5. Using Pre-Trained Model:
If you want to use the already-trained model:

from tensorflow import keras
model = keras.models.load_model('Models/MobileNetV2/plant_disease_mobilenet.keras')


6. My Role Description:

    -Implemented the MobileNetV2 model,Prepared the training pipeline,Trained and fine-tuned the model,Generated evaluation metrics and pushed all files/results to GitHub

---------------------------------------------------------------
Models Evaluation
 
    evaluates and compares three deep learning architectures for plant disease classification: EfficientNetB3, ResNet50, and MobileNet.
The evaluation includes quantitative metrics, qualitative analysis, and explainability results.
1. Evaluation Metrics

Each model was evaluated using the following metrics:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

Classification Report


Model	Accuracy	Precision	Recall	F1-score
0	EfficientNetB3	0.878049	0.900071	0.865741	0.852623
1	ResNet50	0.695122	0.807359	0.711700	0.697907
2	MobileNet	0.731707	0.707211	0.732435	0.684139

Confusion Matrix Analysis

Confusion matrices were generated for all three models to analyze:

Correct vs. incorrect predictions

Which classes are most frequently confused

Whether additional data is needed for specific plant diseases
# ResNet50 – Plant Disease Classification  
### Deep Learning Model Documentation

---

## 1. Model Overview

ResNet50 is a 50-layer deep convolutional neural network that uses residual learning to overcome the vanishing gradient problem.  
The model is widely used for image classification tasks, especially when fine-tuned on domain-specific datasets such as plant diseases.

### Why ResNet50?
- Excellent feature extraction capability  
- Stable training behavior due to skip connections  
- Performs well on medium-sized image datasets  
- Strong transfer learning support via ImageNet pre-trained weights  

Official Documentation:  
https://keras.io/api/applications/resnet/

---

## 2. How to Train the Model

Run the notebook:

```
Models/ResNet50/02_ResNet50_Training.ipynb
```

This notebook handles:
- Dataset loading  
- Preprocessing and normalization  
- Phase 1 (feature extraction)  
- Phase 2 (fine-tuning)  
- Evaluation and saving the final model  

Pre-trained model link (optional):
*Add Google Drive link here*

---

## 3. Dependencies

Install the following packages:

```
TensorFlow
Keras
NumPy
Matplotlib
```

---

## 4. Model Training Pipeline

### Phase 1 — Feature Extraction
- Freeze all ResNet50 layers  
- Train only the newly added classifier head  
- Fast convergence and stable training

### Phase 2 — Fine-Tuning
- Unfreeze the last 30 layers  
- Reduce learning rate  
- Allow the backbone to adapt to plant disease features  

---

## 5. Model Performance

| Metric | Value |
|--------|-------|
| Training Accuracy | ~96–99% |
| Validation Accuracy | ~98–99% |
| Test Accuracy | ~97–99% |

Notes:
- Fine-tuning improved performance significantly  
- Low validation loss indicates strong generalization  

---

## 6. Using the Pre-Trained Model

```
from tensorflow import keras
model = keras.models.load_model("Models/ResNet50/resnet50_final.h5")
```

---

## 7. My Role

- Implemented the entire ResNet50 training pipeline  
- Built and fine-tuned the model  
- Prepared preprocessing and data generators  
- Evaluated model performance and metrics  
- Saved trained weights and uploaded all results to GitHub  

---

## 8. Summary

ResNet50 is a highly reliable architecture for plant disease classification.  
After fine-tuning, it achieved excellent performance and generalization.  
It forms one of the main models in our evaluation, alongside EfficientNetB3 and MobileNetV2.
