# plant-disease-classification
project

# EfficientNetB3
1. Model Architecture:
    -why choose EfficientNetB3? Because it achieves high accuracy with significantly lower computational cost compared to older models
    
    -EfficientNet, with its compound scaling methodology, had an impact on our understanding of the balance between efficiency and accuracy in deep learning. By intelligently scaling width, depth, and resolution, it offers versatile models adaptable to various hardware constraints.
    
    -source: https://blog.roboflow.com/what-is-efficientnet/
    
2. How To Train The Model:
    -run efficientnetb3_training.ipynb

3. Dependencies For Model Training:
    -TensorFlow, Keras, Matplotlib.

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