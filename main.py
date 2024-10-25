import librosa
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
import warnings
import joblib

class AudioEmotionAnalyzer:
    def __init__(self, model_path=None):
        """
        Initialize the Audio Emotion Analyzer with optional pre-trained model.
        
        Args:
            model_path (str, optional): Path to pre-trained model
        """
        self.emotions = ['angry', 'happy', 'sad', 'neutral', 'fear']
        self.sample_rate = 22050
        self.duration = 3  # seconds
        self.feature_scaler = StandardScaler()
        
        # Initialize or load pre-trained model
        if model_path:
            self.model = tf.keras.models.load_model(model_path)
        else:
            self._build_model()
    
    def _build_model(self):
        """Build and compile the emotion classification model."""
        input_shape = (173, 40)  # Mel spectrogram features shape
        
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv1D(64, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.2),
            
            layers.Conv1D(128, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.2),
            
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            layers.Bidirectional(layers.LSTM(32)),
            
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(len(self.emotions), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model

    def extract_features(self, audio_path):
        """
        Extract comprehensive audio features from the audio file.
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            dict: Dictionary containing extracted features
        """
        # Load audio file
        try:
            y, sr = librosa.load(audio_path, duration=self.duration, sr=self.sample_rate)
        except Exception as e:
            raise ValueError(f"Error loading audio file: {str(e)}")
        
        # Feature extraction
        features = {}
        
        # Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        features['mel_spectrogram'] = mel_spec_db
        
        # MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features['mfccs'] = mfccs
        features['mfcc_means'] = np.mean(mfccs, axis=1)
        features['mfcc_vars'] = np.var(mfccs, axis=1)
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features['spectral_centroids_mean'] = np.mean(spectral_centroids)
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        
        # Rhythm features
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = tempo
        
        # Energy features
        features['rms_energy'] = np.mean(librosa.feature.rms(y=y)[0])
        
        # Statistical features
        features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(y))
        features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        
        return features

    def preprocess_features(self, features):
        """
        Preprocess extracted features for model input.
        
        Args:
            features (dict): Dictionary of extracted features
            
        Returns:
            numpy.ndarray: Preprocessed features ready for model input
        """
        # Prepare mel spectrogram for CNN input
        mel_spec = features['mel_spectrogram']
        
        # Ensure consistent shape
        if mel_spec.shape[1] < 173:
            pad_width = ((0, 0), (0, 173 - mel_spec.shape[1]))
            mel_spec = np.pad(mel_spec, pad_width, mode='constant')
        elif mel_spec.shape[1] > 173:
            mel_spec = mel_spec[:, :173]
        
        # Normalize
        mel_spec = (mel_spec - np.mean(mel_spec)) / np.std(mel_spec)
        
        return mel_spec

    def analyze_emotion(self, audio_path, return_probabilities=False):
        """
        Analyze the emotion in an audio file.
        
        Args:
            audio_path (str): Path to audio file
            return_probabilities (bool): Whether to return probability distribution
            
        Returns:
            str: Predicted emotion
            dict: Probability distribution (if return_probabilities=True)
        """
        # Extract features
        features = self.extract_features(audio_path)
        
        # Preprocess features
        processed_features = self.preprocess_features(features)
        
        # Reshape for model input
        model_input = processed_features.reshape(1, 173, 40)
        
        # Get predictions
        predictions = self.model.predict(model_input, verbose=0)
        
        # Get predicted emotion
        predicted_emotion = self.emotions[np.argmax(predictions[0])]
        
        if return_probabilities:
            # Create probability distribution dictionary
            prob_dist = {emotion: float(prob) for emotion, prob in zip(self.emotions, predictions[0])}
            return predicted_emotion, prob_dist
        
        return predicted_emotion

    def analyze_emotion_with_confidence(self, audio_path, confidence_threshold=0.5):
        """
        Analyze emotion with confidence threshold.
        
        Args:
            audio_path (str): Path to audio file
            confidence_threshold (float): Minimum confidence threshold
            
        Returns:
            tuple: (predicted_emotion, confidence, detailed_analysis)
        """
        emotion, probs = self.analyze_emotion(audio_path, return_probabilities=True)
        confidence = probs[emotion]
        
        detailed_analysis = {
            'predicted_emotion': emotion,
            'confidence': confidence,
            'meets_threshold': confidence >= confidence_threshold,
            'probability_distribution': probs,
            'alternative_emotions': sorted(
                [(e, p) for e, p in probs.items() if e != emotion],
                key=lambda x: x[1],
                reverse=True
            )
        }
        
        return emotion, confidence, detailed_analysis

    def save_model(self, path):
        """Save the model to disk."""
        self.model.save(path)

def main():
    """Example usage of the AudioEmotionAnalyzer class."""
    # Initialize analyzer
    analyzer = AudioEmotionAnalyzer()
    
    # Example audio file path
    audio_path = "example_audio.wav"
    
    try:
        # Get detailed analysis
        emotion, confidence, analysis = analyzer.analyze_emotion_with_confidence(
            audio_path,
            confidence_threshold=0.6
        )
        
        print(f"Predicted Emotion: {emotion}")
        print(f"Confidence: {confidence:.2f}")
        print("\nDetailed Analysis:")
        for key, value in analysis.items():
            print(f"{key}: {value}")
            
    except Exception as e:
        print(f"Error analyzing audio: {str(e)}")

if __name__ == "__main__":
    main()
