# Facial-Emotion-Recognition-Using-Deep-Learning

### Project Summary
This project, titled "DeepFER" (Deep Facial Emotion Recognition), is a deep learning-based system designed to recognize human emotions from facial expressions in real-time. It leverages the power of Convolutional Neural Networks (CNNs) and Transfer Learning to classify facial images into one of seven fundamental emotion categories: Angry, Sad, Happy, Fear, Neutral, Disgust, and Surprise.

The system is built with robustness in mind, utilizing a diverse dataset of both posed and spontaneous expressions. To enhance model generalization and accuracy, various data augmentation techniques such as rotation, flipping, and zooming are applied during the data preprocessing phase.

The ultimate goal of DeepFER is to enable practical applications in a wide range of fields, including mental health monitoring, customer service feedback analysis, and advanced human-computer interaction, where the ability to accurately recognize emotions can significantly improve user experience and system responsiveness.

### Features
- Real-time Recognition: Uses a live webcam feed to perform real-time emotion classification.

- Multiple Models: Includes several pre-trained models, such as custom CNNs and a transfer-learned EfficientNet model, allowing for flexible use and experimentation.

- Streamlit Web Interface: A user-friendly web application built with Streamlit provides an easy-to-use interface for demonstration and testing.

- Comprehensive File Structure: The repository is well-organized with dedicated folders for original and preprocessed images, making it easy to understand the project's data flow.

- Jupyter Notebooks: An accompanying Jupyter notebook (model_testing.ipynb) details the complete model training, evaluation, and fine-tuning process.

### Technology Stack
- Deep Learning: TensorFlow, Keras

- Frontend: Streamlit

- Image Processing: OpenCV

- Language: Python

### File Structure

- original_images/: Contains the raw, unprocessed images used for training the models.
- preprocessed_images/: Houses the images after they have been processed and augmented, ready for model training.
- app.py: The main script that runs the Streamlit web application for real-time emotion detection.
- best_emotion_model.keras: The final, best-performing model saved in Keras format.
- cnn_custom_first_model.keras: An example of a custom-built Convolutional Neural Network model.
- final_transfer_learning_emotion_recognition_model.keras: A final model trained using the transfer learning approach.
- Transfer_learning_efficientnet_model.keras: A model specifically trained using the EfficientNet architecture via transfer learning.
- model_testing.ipynb: A Jupyter Notebook dedicated to testing and evaluating the different models.
- DeepFER: Facial Emotion Recognition Using Deep Learning.ipynb: The primary Jupyter Notebook for experimentation, data preprocessing, and building the deep learning models from scratch.
- requirements.txt: A comprehensive list of all the Python libraries and their versions required to run the project.
- README.md: The main README file for the project.

### Contribution
Individual Contributor: Mritunjay Mishra .
