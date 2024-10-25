import os
import numpy as np
import pandas as pd
import tensorflow as tf
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import augment_audio
from datetime import datetime
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import warnings
import logging

class EmotionDataGenerator(tf.keras.utils.Sequence):
    """Custom data generator for emotion recognition training."""
    
    def __init__(self, file_paths, labels, batch_size=32, dim=(173, 40), 
                 n_channels=1, n_classes=5, shuffle=True, augment=False):
        """Initialize the data generator."""
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.file_paths = file_paths
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment = augment
        self.on_epoch_end()
        
        # Initialize augmentation parameters
        self.augment_params = {
            'pitch_shift_range': (-2, 2),
            'time_stretch_range': (0.8, 1.2),
            'noise_factor_range': (0.001, 0.005)
        }
    
    def __len__(self):
        """Calculate number of batches per epoch."""
        return int(np.floor(len(self.file_paths) / self.batch_size))
    
    def __getitem__(self, index):
        """Generate one batch of data."""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        
        # Find list of IDs
        file_paths_temp = [self.file_paths[k] for k in indexes]
        labels_temp = [self.labels[k] for k in indexes]
        
        # Generate data
        X, y = self.__data_generation(file_paths_temp, labels_temp)
        
        return X, y
    
    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        self.indexes = np.arange(len(self.file_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, file_paths_temp, labels_temp):
        """Generate data containing batch_size samples."""
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, self.n_classes), dtype=int)
        
        # Generate data
        for i, (path, label) in enumerate(zip(file_paths_temp, labels_temp)):
            # Load and preprocess audio
            features = self._load_and_preprocess_audio(path)
            
            if self.augment and random.random() < 0.5:
                features = self._augment_features(features)
            
            X[i,] = features.reshape((*self.dim, self.n_channels))
            y[i,] = label
            
        return X, y
    
    def _load_and_preprocess_audio(self, file_path):
        """Load and preprocess audio file."""
        try:
            # Load audio file
            y, sr = librosa.load(file_path, duration=3)
            
            # Extract mel spectrogram
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Ensure consistent shape
            if mel_spec_db.shape[1] < self.dim[0]:
                pad_width = ((0, 0), (0, self.dim[0] - mel_spec_db.shape[1]))
                mel_spec_db = np.pad(mel_spec_db, pad_width, mode='constant')
            else:
                mel_spec_db = mel_spec_db[:, :self.dim[0]]
            
            # Normalize
            mel_spec_db = (mel_spec_db - np.mean(mel_spec_db)) / np.std(mel_spec_db)
            
            return mel_spec_db
            
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {str(e)}")
            return np.zeros(self.dim)
    
    def _augment_features(self, features):
        """Apply audio augmentation to features."""
        # Random augmentation selection
        augmentation_type = random.choice(['pitch', 'stretch', 'noise', 'none'])
        
        if augmentation_type == 'pitch':
            pitch_shift = random.uniform(*self.augment_params['pitch_shift_range'])
            features = librosa.effects.pitch_shift(features, sr=22050, n_steps=pitch_shift)
        elif augmentation_type == 'stretch':
            stretch_factor = random.uniform(*self.augment_params['time_stretch_range'])
            features = librosa.effects.time_stretch(features, rate=stretch_factor)
        elif augmentation_type == 'noise':
            noise_factor = random.uniform(*self.augment_params['noise_factor_range'])
            noise = np.random.normal(0, 1, features.shape)
            features = features + noise_factor * noise
            
        return features

class EmotionModelTrainer:
    """Class for training emotion recognition models."""
    
    def __init__(self, model_config, training_config):
        """Initialize the trainer with configurations."""
        self.model_config = model_config
        self.training_config = training_config
        self.history = None
        self.model = None
        self.setup_logging()
    
    def setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training.log'),
                logging.StreamHandler()
            ]
        )
    
    def prepare_data(self, data_path):
        """Prepare training, validation, and test datasets."""
        # Load data paths and labels
        df = pd.read_csv(os.path.join(data_path, 'metadata.csv'))
        
        # Encode labels
        le = LabelEncoder()
        encoded_labels = le.fit_transform(df['emotion'])
        one_hot_labels = tf.keras.utils.to_categorical(encoded_labels)
        
        # Split data
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            df['file_path'], one_hot_labels, 
            test_size=0.2, stratify=encoded_labels
        )
        
        # Create data generators
        train_generator = EmotionDataGenerator(
            train_paths.values, train_labels,
            batch_size=self.training_config['batch_size'],
            augment=True
        )
        
        val_generator = EmotionDataGenerator(
            val_paths.values, val_labels,
            batch_size=self.training_config['batch_size'],
            augment=False
        )
        
        return train_generator, val_generator, le
    
    def build_model(self):
        """Build and compile the model."""
        input_shape = (173, 40, 1)
        
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            
            # CNN layers
            tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.3),
            
            # Reshape for LSTM
            tf.keras.layers.Reshape((-1, 256)),
            
            # LSTM layers
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            
            # Dense layers
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(5, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.training_config['learning_rate']),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, train_generator, val_generator):
        """Train the model."""
        # Create callbacks
        callbacks = [
            ModelCheckpoint(
                'best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=self.training_config['epochs'],
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, test_generator):
        """Evaluate the model."""
        results = self.model.evaluate(test_generator)
        return dict(zip(self.model.metrics_names, results))
    
    def save_training_results(self):
        """Save training history and results."""
        if self.history is None:
            logging.warning("No training history available to save.")
            return
        
        # Create results directory
        results_dir = 'training_results'
        os.makedirs(results_dir, exist_ok=True)
        
        # Save history
        history_dict = {
            'loss': self.history.history['loss'],
            'val_loss': self.history.history['val_loss'],
            'accuracy': self.history.history['accuracy'],
            'val_accuracy': self.history.history['val_accuracy']
        }
        
        with open(os.path.join(results_dir, 'training_history.json'), 'w') as f:
            json.dump(history_dict, f)
        
        # Plot and save training curves
        self.plot_training_curves(results_dir)
    
    def plot_training_curves(self, save_dir):
        """Plot and save training curves."""
        plt.figure(figsize=(12, 4))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'))
        plt.close()

def main():
    """Main training script."""
    # Configuration
    model_config = {
        'input_shape': (173, 40, 1),
        'n_classes': 5
    }
    
    training_config = {
        'batch_size': 32,
        'epochs': 100,
        'learning_rate': 0.001,
        'data_path': 'path/to/dataset'
    }
    
    try:
        # Initialize trainer
        trainer = EmotionModelTrainer(model_config, training_config)
        
        # Prepare data
        train_generator, val_generator, label_encoder = trainer.prepare_data(
            training_config['data_path']
        )
        
        # Build and train model
        trainer.build_model()
        trainer.train(train_generator, val_generator)
        
        # Save results
        trainer.save_training_results()
        
        logging.info("Training completed successfully!")
        
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()
