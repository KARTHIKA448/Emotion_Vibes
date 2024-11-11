### Emotion Vibes: A Speech-Based Emotion Detection Web App
Welcome to Emotion Vibes, a powerful web app designed to detect emotions from speech and audio inputs. Leveraging state-of-the-art machine learning techniques and the RAVDESS dataset, this project aims to provide real-time emotion classification for various speech emotions. The web app is built using Django for the backend and incorporates advanced Deep Learning algorithms for accurate emotion recognition.

Emotion Vibes offers an easy-to-use platform to classify emotions from audio files. By analyzing speech patterns, the app can identify a variety of emotions such as:
- Calm
- Happy
- Fearful
- Disgust
The system uses `MLP (Multilayer Perceptron)` Classifiers for machine learning and Librosa for extracting audio features like MFCC, Chroma, and Mel spectrograms.

### 🌟 Key Features
- Emotion Recognition: Detect emotions from audio files such as calm, happy, fearful, and disgust.
- Real-Time Predictions: Upload your own audio files and get real-time emotion predictions.
- Interactive UI: Designed with Django, the web app provides a user-friendly interface to interact with the model.
- Visualization: Visual output of prediction results, including accuracy and confusion matrices.

### 🛠️ Technologies Used
- Django: Web framework for building the application.
Python Libraries:
  - Librosa: Audio feature extraction.
  - Soundfile: Reading and processing audio files.
  - NumPy: Array manipulations for feature processing.
  - Scikit-learn: Machine learning (MLP Classifier).
  - Matplotlib: Data visualization for model performance.
  - Machine Learning Models: MLP Classifier for emotion classification.
- HTML/CSS: For frontend design.
### 📊 Dataset
The project uses the `RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)` dataset. The dataset contains labeled audio samples representing a range of human emotions. For this app, we have focused on four primary emotions:
  Calm
  Happy
  Fearful
  Disgust

### 💻 Model Training
The model is trained using an MLP Classifier from sklearn.neural_network.

### Feature Extraction:
A custom function extract_feature() extracts audio features (MFCCs, chroma, and mel spectrograms) using Librosa. These features are used to train the model. We utilize the RAVDESS dataset, focusing on emotions like calm, happy, fearful, and disgust.

### Data Preparation:
The dataset is split into training and testing sets (75% training, 25% testing). The features are organized into a matrix, where each row represents an audio sample, and each column represents an extracted feature (180 features in total).

### MLP Classifier & Hyperparameter Tuning:
An MLPClassifier model is trained using GridSearchCV for hyperparameter tuning. The hyperparameter grid includes configurations for hidden layer sizes, activation functions, solvers, learning rates, batch sizes, and iteration limits. The best parameters from the grid search are:

    Hidden layers: (200, 200)
    Activation: Tanh
    Solver: Adam
    Batch size: 256
    Alpha: 0.01
    Learning rate: Constant
    The best cross-validation accuracy achieved is 78.3%.

### Model Training & Evaluation:
The model achieves:

      Training Accuracy: 94.44%
      Test Accuracy: 73.44%
A learning curve plot visualizes the model’s loss over training iterations.

### Model Deployment:
The optimized model is saved using pickle and deployed in a Django web app. Users can upload audio files, and the model will predict the emotion conveyed in the audio. The app also displays corresponding GIFs based on the predicted emotion for an enhanced experience.

### Emotion Vibes Django app

    - Upload an audio file (WAV format) via the web interface.
    - The model will process the audio and predict the emotion.
    - Results, including the detected emotion, will be displayed on the webpage.

###### HOME PAGE
![](https://github.com/KARTHIKA448/Emotion_Vibes/blob/main/outputs/home.png)
###### UPLOADING FILES
![](https://github.com/KARTHIKA448/Emotion_Vibes/blob/main/outputs/uploadfile.png)
![](https://github.com/KARTHIKA448/Emotion_Vibes/blob/main/outputs/file uploaded.png)

###### PREDICTED EMOTIONS

`CALM` 
`HAPPY`
`FEARFUL`
`DISGUST'

    
