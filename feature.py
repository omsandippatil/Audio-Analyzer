import librosa
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks
import pyworld as world
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
import joblib
import threading
import queue
import soundfile as sf
from datetime import datetime
import json

class AdvancedAudioFeatures:
    def __init__(self):
        """Initialize the advanced audio feature analyzer."""
        self.sample_rate = 22050
        self.frame_length = 2048
        self.hop_length = 512
        self.analysis_cache = {}
        
    def extract_prosodic_features(self, y, sr):
        """
        Extract prosodic features including pitch, energy, and rhythm patterns.
        
        Args:
            y (numpy.ndarray): Audio signal
            sr (int): Sample rate
            
        Returns:
            dict: Prosodic features
        """
        features = {}
        
        # Pitch features using WORLD vocoder
        _f0, t = librosa.piptrack(y=y, sr=sr)
        f0 = np.mean(_f0, axis=0)
        features['pitch_mean'] = np.mean(f0[f0 > 0])
        features['pitch_std'] = np.std(f0[f0 > 0])
        features['pitch_range'] = np.ptp(f0[f0 > 0])
        
        # Speech rate estimation
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        features['speech_rate'] = tempo
        
        # Energy contour
        rms = librosa.feature.rms(y=y)[0]
        features['energy_mean'] = np.mean(rms)
        features['energy_std'] = np.std(rms)
        features['energy_range'] = np.ptp(rms)
        
        return features
    
    def extract_voice_quality_features(self, y, sr):
        """
        Extract voice quality features including jitter, shimmer, and harmonicity.
        
        Args:
            y (numpy.ndarray): Audio signal
            sr (int): Sample rate
            
        Returns:
            dict: Voice quality features
        """
        features = {}
        
        # Convert to mono if stereo
        if len(y.shape) > 1:
            y = librosa.to_mono(y)
            
        # Harmonics-to-noise ratio
        harmonics = librosa.effects.harmonic(y)
        residual = librosa.effects.percussive(y)
        features['hnr'] = np.mean(np.abs(harmonics)) / np.mean(np.abs(residual))
        
        # Jitter (pitch perturbation)
        f0, voiced_flag, _ = librosa.pyin(y, 
                                        fmin=librosa.note_to_hz('C2'),
                                        fmax=librosa.note_to_hz('C7'))
        if voiced_flag.any():
            voiced_f0 = f0[voiced_flag]
            jitter = np.mean(np.abs(np.diff(voiced_f0))) / np.mean(voiced_f0)
            features['jitter'] = jitter
        else:
            features['jitter'] = 0
            
        # Shimmer (amplitude perturbation)
        amplitude_env = np.abs(librosa.stft(y))
        shimmer = np.mean(np.abs(np.diff(amplitude_env, axis=1))) / np.mean(amplitude_env)
        features['shimmer'] = np.mean(shimmer)
        
        return features
    
    def extract_spectral_features(self, y, sr):
        """
        Extract advanced spectral features.
        
        Args:
            y (numpy.ndarray): Audio signal
            sr (int): Sample rate
            
        Returns:
            dict: Spectral features
        """
        features = {}
        
        # Spectral features
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        # Statistical measures of spectral features
        features.update({
            'spectral_centroid_mean': np.mean(spec_cent),
            'spectral_centroid_std': np.std(spec_cent),
            'spectral_bandwidth_mean': np.mean(spec_bw),
            'spectral_bandwidth_std': np.std(spec_bw),
            'spectral_rolloff_mean': np.mean(spec_rolloff),
            'spectral_rolloff_std': np.std(spec_rolloff),
            'spectral_contrast_mean': np.mean(spec_contrast),
            'spectral_contrast_std': np.std(spec_contrast)
        })
        
        # Mel-frequency features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        
        # Statistical measures of MFCCs
        for i in range(20):
            features[f'mfcc{i+1}_mean'] = np.mean(mfcc[i])
            features[f'mfcc{i+1}_std'] = np.std(mfcc[i])
            features[f'delta_mfcc{i+1}_mean'] = np.mean(delta_mfcc[i])
            features[f'delta2_mfcc{i+1}_mean'] = np.mean(delta2_mfcc[i])
            
        return features
    
    def extract_emotional_features(self, y, sr):
        """
        Extract features specifically relevant to emotion detection.
        
        Args:
            y (numpy.ndarray): Audio signal
            sr (int): Sample rate
            
        Returns:
            dict: Emotional features
        """
        features = {}
        
        # Energy features
        energy = np.sum(np.abs(y)**2)
        features['energy'] = energy
        
        # Pitch variability
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_mean = np.mean(pitches[pitches > 0])
        pitch_std = np.std(pitches[pitches > 0])
        features['pitch_variability'] = pitch_std / pitch_mean if pitch_mean > 0 else 0
        
        # Speech rate variation
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        peaks = find_peaks(onset_env)[0]
        if len(peaks) > 1:
            features['speech_rate_variation'] = np.std(np.diff(peaks))
        else:
            features['speech_rate_variation'] = 0
            
        # Emotional stress indicators
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features['harmonic_complexity'] = np.std(np.mean(chroma, axis=1))
        
        return features
    
    def analyze_audio_segments(self, audio_path, segment_duration=3.0):
        """
        Analyze audio in segments for temporal emotion tracking.
        
        Args:
            audio_path (str): Path to audio file
            segment_duration (float): Duration of each segment in seconds
            
        Returns:
            list: List of segment analyses
        """
        y, sr = librosa.load(audio_path)
        segment_samples = int(segment_duration * sr)
        segments = []
        
        for i in range(0, len(y), segment_samples):
            segment = y[i:i + segment_samples]
            if len(segment) >= segment_samples // 2:  # Analyze if segment is at least half the desired duration
                analysis = {
                    'start_time': i / sr,
                    'end_time': min((i + segment_samples) / sr, len(y) / sr),
                    'prosodic': self.extract_prosodic_features(segment, sr),
                    'voice_quality': self.extract_voice_quality_features(segment, sr),
                    'spectral': self.extract_spectral_features(segment, sr),
                    'emotional': self.extract_emotional_features(segment, sr)
                }
                segments.append(analysis)
                
        return segments
    
    def generate_feature_summary(self, segments):
        """
        Generate a summary of features across segments.
        
        Args:
            segments (list): List of segment analyses
            
        Returns:
            dict: Summary statistics
        """
        summary = {
            'temporal_evolution': {},
            'overall_statistics': {},
            'emotional_indicators': {}
        }
        
        # Extract temporal evolution
        for feature_type in ['prosodic', 'voice_quality', 'spectral', 'emotional']:
            feature_values = {}
            for segment in segments:
                for feature, value in segment[feature_type].items():
                    if feature not in feature_values:
                        feature_values[feature] = []
                    feature_values[feature].append(value)
                    
            # Calculate statistics for each feature
            for feature, values in feature_values.items():
                summary['temporal_evolution'][feature] = {
                    'trend': np.polyfit(range(len(values)), values, 1)[0],
                    'variability': np.std(values),
                    'range': np.ptp(values)
                }
                
        # Calculate overall statistics
        all_features = {}
        for segment in segments:
            for feature_type, features in segment.items():
                if isinstance(features, dict):
                    for feature, value in features.items():
                        if feature not in all_features:
                            all_features[feature] = []
                        all_features[feature].append(value)
                        
        for feature, values in all_features.items():
            summary['overall_statistics'][feature] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'skewness': skew(values),
                'kurtosis': kurtosis(values)
            }
            
        return summary
    
    def save_analysis(self, analysis, output_path):
        """
        Save analysis results to a file.
        
        Args:
            analysis (dict): Analysis results
            output_path (str): Path to save the analysis
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"audio_analysis_{timestamp}.json"
        full_path = f"{output_path}/{filename}"
        
        # Convert numpy values to Python native types
        def convert_to_native(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            return obj
        
        analysis = convert_to_native(analysis)
        
        with open(full_path, 'w') as f:
            json.dump(analysis, f, indent=2)
            
        return full_path

def main():
    """Example usage of the AdvancedAudioFeatures class."""
    analyzer = AdvancedAudioFeatures()
    
    try:
        # Example audio file
        audio_path = "example_audio.wav"
        
        # Perform segmented analysis
        print("Analyzing audio segments...")
        segments = analyzer.analyze_audio_segments(audio_path)
        
        # Generate summary
        print("Generating feature summary...")
        summary = analyzer.generate_feature_summary(segments)
        
        # Save analysis
        output_path = "analysis_output"
        saved_path = analyzer.save_analysis({
            'segments': segments,
            'summary': summary
        }, output_path)
        
        print(f"Analysis saved to: {saved_path}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()
