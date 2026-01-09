import numpy as np
from scipy import stats, signal
from scipy.fft import fft, fftfreq
import pandas as pd
from tqdm import tqdm

class TimeSeriesFeatureExtractor:
    """
    Extractor de características estadísticas, temporales y frecuenciales
    para secuencias temporales multivariadas.
    """
    
    def __init__(self, sampling_rate=100):
        """
        Args:
            sampling_rate: Frecuencia de muestreo en Hz (para análisis frecuencial)
        """
        self.sampling_rate = sampling_rate
        self.feature_names = []
    
    def extract_statistical_features(self, sequence, channel_idx):
        """
        Extrae características estadísticas básicas de una secuencia.
        
        Args:
            sequence: Array de forma (time_steps, n_channels)
            channel_idx: Índice del canal a procesar
        
        Returns:
            dict: Características estadísticas
        """
        data = sequence[:, channel_idx]
        
        features = {
            f'ch{channel_idx}_mean': np.mean(data),
            f'ch{channel_idx}_std': np.std(data),
            f'ch{channel_idx}_min': np.min(data),
            f'ch{channel_idx}_max': np.max(data),
            f'ch{channel_idx}_median': np.median(data),
            f'ch{channel_idx}_q25': np.percentile(data, 25),
            f'ch{channel_idx}_q75': np.percentile(data, 75),
            f'ch{channel_idx}_iqr': np.percentile(data, 75) - np.percentile(data, 25),
            f'ch{channel_idx}_range': np.max(data) - np.min(data),
            f'ch{channel_idx}_skewness': stats.skew(data),
            f'ch{channel_idx}_kurtosis': stats.kurtosis(data),
            f'ch{channel_idx}_rms': np.sqrt(np.mean(data**2)),
        }
        
        return features
    
    def extract_temporal_features(self, sequence, channel_idx):
        """
        Extrae características temporales (cambios, cruces por cero, etc.).
        
        Args:
            sequence: Array de forma (time_steps, n_channels)
            channel_idx: Índice del canal
        
        Returns:
            dict: Características temporales
        """
        data = sequence[:, channel_idx]
        
        # Diferencias y velocidad de cambio
        diff = np.diff(data)
        
        # Cruces por cero
        zero_crossings = np.sum(np.diff(np.sign(data)) != 0)
        
        # Cruces por la media
        mean_crossings = np.sum(np.diff(np.sign(data - np.mean(data))) != 0)
        
        # Autocorrelación (lag=1)
        autocorr_lag1 = np.corrcoef(data[:-1], data[1:])[0, 1] if len(data) > 1 else 0
        
        features = {
            f'ch{channel_idx}_mean_abs_diff': np.mean(np.abs(diff)),
            f'ch{channel_idx}_std_diff': np.std(diff),
            f'ch{channel_idx}_max_abs_diff': np.max(np.abs(diff)),
            f'ch{channel_idx}_zero_crossings': zero_crossings,
            f'ch{channel_idx}_mean_crossings': mean_crossings,
            f'ch{channel_idx}_autocorr_lag1': autocorr_lag1,
            f'ch{channel_idx}_trend': stats.linregress(np.arange(len(data)), data)[0],  # Pendiente
        }
        
        return features
    
    def extract_frequency_features(self, sequence, channel_idx, n_freq_bins=5):
        """
        Extrae características frecuenciales usando FFT.
        
        Args:
            sequence: Array de forma (time_steps, n_channels)
            channel_idx: Índice del canal
            n_freq_bins: Número de bins de frecuencia a extraer
        
        Returns:
            dict: Características frecuenciales
        """
        data = sequence[:, channel_idx]
        n = len(data)
        
        # FFT
        fft_vals = fft(data)
        fft_freqs = fftfreq(n, 1/self.sampling_rate)
        
        # Usar solo frecuencias positivas
        positive_freqs = fft_freqs[:n//2]
        fft_magnitude = np.abs(fft_vals[:n//2])
        
        # Potencia espectral
        power_spectrum = fft_magnitude ** 2
        total_power = np.sum(power_spectrum)
        
        features = {
            f'ch{channel_idx}_spectral_centroid': np.sum(positive_freqs * power_spectrum) / (total_power + 1e-10),
            f'ch{channel_idx}_spectral_spread': np.sqrt(np.sum(((positive_freqs - features.get(f'ch{channel_idx}_spectral_centroid', 0))**2) * power_spectrum) / (total_power + 1e-10)),
            f'ch{channel_idx}_spectral_entropy': stats.entropy(power_spectrum + 1e-10),
            f'ch{channel_idx}_dominant_freq': positive_freqs[np.argmax(fft_magnitude)] if len(fft_magnitude) > 0 else 0,
            f'ch{channel_idx}_total_power': total_power,
        }
        
        # Potencia en bandas frecuenciales
        freq_bands = np.linspace(0, self.sampling_rate/2, n_freq_bins + 1)
        for i in range(n_freq_bins):
            band_mask = (positive_freqs >= freq_bands[i]) & (positive_freqs < freq_bands[i+1])
            band_power = np.sum(power_spectrum[band_mask])
            features[f'ch{channel_idx}_band{i}_power'] = band_power
            features[f'ch{channel_idx}_band{i}_ratio'] = band_power / (total_power + 1e-10)
        
        return features
    
    def extract_multichannel_features(self, sequence):
        """
        Extrae características de interacción entre canales.
        
        Args:
            sequence: Array de forma (time_steps, n_channels)
        
        Returns:
            dict: Características multicanal
        """
        n_channels = sequence.shape[1]
        features = {}
        
        # Correlaciones entre canales (solo algunos pares para no explotar dimensionalidad)
        # Tomar máximo 10 pares de correlación
        max_pairs = min(10, (n_channels * (n_channels - 1)) // 2)
        pair_count = 0
        
        for i in range(n_channels):
            for j in range(i+1, n_channels):
                if pair_count >= max_pairs:
                    break
                
                corr = np.corrcoef(sequence[:, i], sequence[:, j])[0, 1]
                features[f'corr_ch{i}_ch{j}'] = corr
                pair_count += 1
            
            if pair_count >= max_pairs:
                break
        
        # Estadísticas globales
        features['global_mean'] = np.mean(sequence)
        features['global_std'] = np.std(sequence)
        features['global_max'] = np.max(sequence)
        features['global_min'] = np.min(sequence)
        
        return features
    
    def extract_all_features(self, sequence):
        """
        Extrae todas las características de una secuencia.
        
        Args:
            sequence: Array de forma (time_steps, n_channels)
        
        Returns:
            dict: Todas las características
        """
        all_features = {}
        n_channels = sequence.shape[1]
        
        # Características por canal
        for ch in range(n_channels):
            # Estadísticas
            all_features.update(self.extract_statistical_features(sequence, ch))
            # Temporales
            all_features.update(self.extract_temporal_features(sequence, ch))
            # Frecuenciales
            all_features.update(self.extract_frequency_features(sequence, ch))
        
        # Características multicanal
        all_features.update(self.extract_multichannel_features(sequence))
        
        return all_features
    
    def transform(self, sequences, verbose=True):
        """
        Transforma un conjunto de secuencias en features tabulares.
        
        Args:
            sequences: Array de forma (n_samples, time_steps, n_channels)
            verbose: Mostrar barra de progreso
        
        Returns:
            pd.DataFrame: DataFrame con features extraídas (n_samples, n_features)
        """
        feature_list = []
        
        iterator = tqdm(sequences, desc="Extrayendo features") if verbose else sequences
        
        for seq in iterator:
            features = self.extract_all_features(seq)
            feature_list.append(features)
        
        df = pd.DataFrame(feature_list)
        
        # Guardar nombres de features
        self.feature_names = df.columns.tolist()
        
        # Reemplazar NaN/Inf por valores válidos
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        return df


def create_tabular_dataset(feature_matrices, labels, subject_ids, extractor):
    """
    Crea un dataset tabular con features extraídas.
    
    Args:
        feature_matrices: Array (n_samples, time_steps, n_channels)
        labels: Array (n_samples,) con etiquetas
        subject_ids: Array (n_samples,) con IDs de sujetos
        extractor: TimeSeriesFeatureExtractor
    
    Returns:
        pd.DataFrame: Dataset tabular con columnas adicionales de metadata
    """
    print("\n" + "="*60)
    print("Creando dataset tabular con feature engineering")
    print("="*60)
    
    # Extraer features
    X_tabular = extractor.transform(feature_matrices, verbose=True)
    
    # Agregar metadata
    X_tabular['subject_id'] = subject_ids
    X_tabular['label'] = labels
    
    print(f"\n✓ Dataset tabular creado:")
    print(f"  - Muestras: {len(X_tabular)}")
    print(f"  - Features: {len(extractor.feature_names)}")
    print(f"  - Sujetos únicos: {len(np.unique(subject_ids))}")
    print(f"  - Clases: {np.unique(labels)}")
    
    return X_tabular
