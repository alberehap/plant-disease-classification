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