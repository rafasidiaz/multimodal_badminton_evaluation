"""
Pipeline Completo XGBoost con Feature Engineering y Nested LOSO
Todo en un solo archivo para facilitar el uso.
"""

import numpy as np
import pandas as pd
import h5py
import os
from datetime import datetime
from tqdm import tqdm
from scipy import stats, signal
from scipy.fft import fft, fftfreq
import xgboost as xgb
from sklearn.metrics import (balanced_accuracy_score, f1_score, 
                             confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
import json
import optuna

# ============================================================================
# PARTE 1: FEATURE ENGINEERING
# ============================================================================

class TimeSeriesFeatureExtractor:
    """
    Extractor de características estadísticas, temporales y frecuenciales
    para secuencias temporales multivariadas.
    """
    
    def __init__(self, sampling_rate=100):
        self.sampling_rate = sampling_rate
        self.feature_names = []
    
    def extract_statistical_features(self, sequence, channel_idx):
        """Extrae características estadísticas básicas."""
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
        """Extrae características temporales."""
        data = sequence[:, channel_idx]
        diff = np.diff(data)
        zero_crossings = np.sum(np.diff(np.sign(data)) != 0)
        mean_crossings = np.sum(np.diff(np.sign(data - np.mean(data))) != 0)
        autocorr_lag1 = np.corrcoef(data[:-1], data[1:])[0, 1] if len(data) > 1 else 0
        
        features = {
            f'ch{channel_idx}_mean_abs_diff': np.mean(np.abs(diff)),
            f'ch{channel_idx}_std_diff': np.std(diff),
            f'ch{channel_idx}_max_abs_diff': np.max(np.abs(diff)),
            f'ch{channel_idx}_zero_crossings': zero_crossings,
            f'ch{channel_idx}_mean_crossings': mean_crossings,
            f'ch{channel_idx}_autocorr_lag1': autocorr_lag1,
            f'ch{channel_idx}_trend': stats.linregress(np.arange(len(data)), data)[0],
        }
        
        return features
    
    def extract_frequency_features(self, sequence, channel_idx, n_freq_bins=5):
        """Extrae características frecuenciales usando FFT."""
        data = sequence[:, channel_idx]
        n = len(data)
        
        fft_vals = fft(data)
        fft_freqs = fftfreq(n, 1/self.sampling_rate)
        
        positive_freqs = fft_freqs[:n//2]
        fft_magnitude = np.abs(fft_vals[:n//2])
        power_spectrum = fft_magnitude ** 2
        total_power = np.sum(power_spectrum)
        
        centroid = np.sum(positive_freqs * power_spectrum) / (total_power + 1e-10)
        
        features = {
            f'ch{channel_idx}_spectral_centroid': centroid,
            f'ch{channel_idx}_spectral_spread': np.sqrt(np.sum(((positive_freqs - centroid)**2) * power_spectrum) / (total_power + 1e-10)),
            f'ch{channel_idx}_spectral_entropy': stats.entropy(power_spectrum + 1e-10),
            f'ch{channel_idx}_dominant_freq': positive_freqs[np.argmax(fft_magnitude)] if len(fft_magnitude) > 0 else 0,
            f'ch{channel_idx}_total_power': total_power,
        }
        
        # Potencia en bandas
        freq_bands = np.linspace(0, self.sampling_rate/2, n_freq_bins + 1)
        for i in range(n_freq_bins):
            band_mask = (positive_freqs >= freq_bands[i]) & (positive_freqs < freq_bands[i+1])
            band_power = np.sum(power_spectrum[band_mask])
            features[f'ch{channel_idx}_band{i}_power'] = band_power
            features[f'ch{channel_idx}_band{i}_ratio'] = band_power / (total_power + 1e-10)
        
        return features
    
    def extract_multichannel_features(self, sequence):
        """Extrae características de interacción entre canales."""
        n_channels = sequence.shape[1]
        features = {}
        
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
        
        features['global_mean'] = np.mean(sequence)
        features['global_std'] = np.std(sequence)
        features['global_max'] = np.max(sequence)
        features['global_min'] = np.min(sequence)
        
        return features
    
    def extract_all_features(self, sequence):
        """Extrae todas las características de una secuencia."""
        all_features = {}
        n_channels = sequence.shape[1]
        
        for ch in range(n_channels):
            all_features.update(self.extract_statistical_features(sequence, ch))
            all_features.update(self.extract_temporal_features(sequence, ch))
            all_features.update(self.extract_frequency_features(sequence, ch))
        
        all_features.update(self.extract_multichannel_features(sequence))
        
        return all_features
    
    def transform(self, sequences, verbose=True):
        """Transforma secuencias en features tabulares."""
        feature_list = []
        iterator = tqdm(sequences, desc="Extrayendo features") if verbose else sequences
        
        for seq in iterator:
            features = self.extract_all_features(seq)
            feature_list.append(features)
        
        df = pd.DataFrame(feature_list)
        self.feature_names = df.columns.tolist()
        
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        return df


# ============================================================================
# PARTE 2: PIPELINE XGBOOST CON LOSO
# ============================================================================

class XGBoostLOSOPipeline:
    """Pipeline completo de XGBoost con Leave-One-Subject-Out."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.results = []
        self.confusion_matrices = []
        self.feature_importance = {}
        
    def nested_loso_splits(self, df, n_val_subjects=1):
        """Genera splits LOSO con validación interna por sujetos."""
        unique_subjects = df['subject_id'].unique()
        
        for test_subject in unique_subjects:
            test_mask = df['subject_id'] == test_subject
            test_df = df[test_mask].copy()
            
            remaining_df = df[~test_mask].copy()
            remaining_subjects = remaining_df['subject_id'].unique()
            
            if len(remaining_subjects) < n_val_subjects + 1:
                train_df = remaining_df
                val_df = remaining_df.sample(frac=0.2, random_state=self.random_state)
            else:
                subject_labels = []
                for subj in remaining_subjects:
                    subj_data = remaining_df[remaining_df['subject_id'] == subj]
                    majority_class = subj_data['label'].mode()[0]
                    subject_labels.append((subj, majority_class))
                
                subject_labels = pd.DataFrame(subject_labels, columns=['subject_id', 'label'])
                
                val_subjects = []
                for label_class in subject_labels['label'].unique():
                    class_subjects = subject_labels[subject_labels['label'] == label_class]['subject_id'].values
                    n_from_class = max(1, n_val_subjects // len(subject_labels['label'].unique()))
                    n_from_class = min(n_from_class, len(class_subjects))
                    
                    selected = np.random.choice(class_subjects, size=n_from_class, replace=False)
                    val_subjects.extend(selected)
                
                val_subjects = val_subjects[:n_val_subjects]
                
                val_df = remaining_df[remaining_df['subject_id'].isin(val_subjects)].copy()
                train_df = remaining_df[~remaining_df['subject_id'].isin(val_subjects)].copy()
            
            yield train_df, val_df, test_df, test_subject
    
    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val, n_trials=50):
        """Optimiza hiperparámetros de XGBoost usando validación."""
        
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                'random_state': self.random_state,
                'objective': 'multi:softmax',
                'num_class': len(np.unique(y_train)),
                'eval_metric': 'mlogloss'
            }
            
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      early_stopping_rounds=20,
                      verbose=False)

            
            y_pred = model.predict(X_val)
            return balanced_accuracy_score(y_val, y_pred)
        
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        return study.best_params
    
    def train_and_evaluate(self, df_tabular, technique, subset, 
                          base_output_folder, optimize=True, n_trials=30):
        """Pipeline completo de entrenamiento y evaluación con LOSO."""
        
        print(f"\n{'='*80}")
        print(f"Entrenamiento XGBoost con LOSO")
        print(f"Técnica: {technique} | Subset: {subset}")
        print(f"{'='*80}\n")
        
        os.makedirs(base_output_folder, exist_ok=True)
        
        feature_cols = [col for col in df_tabular.columns 
                       if col not in ['subject_id', 'label']]
        
        fold_results = []
        all_test_true = []
        all_test_pred = []
        
        splits = list(self.nested_loso_splits(df_tabular, n_val_subjects=2))
        
        # Guardar info de splits
        splits_info = []
        for i, (train_df, val_df, test_df, test_subj) in enumerate(splits):
            splits_info.append({
                'fold': i + 1,
                'test_subject': test_subj,
                'train_subjects': train_df['subject_id'].unique().tolist(),
                'val_subjects': val_df['subject_id'].unique().tolist(),
                'n_train': len(train_df),
                'n_val': len(val_df),
                'n_test': len(test_df)
            })
        
        with open(os.path.join(base_output_folder, 'loso_splits.json'), 'w') as f:
            json.dump(splits_info, f, indent=4)
        
        for fold, (train_df, val_df, test_df, test_subject) in enumerate(splits):
            print(f"\n{'─'*80}")
            print(f"LOSO Fold {fold+1}/{len(splits)} - Test: {test_subject}")
            print(f"  Train: {len(train_df)} samples, Val: {len(val_df)}, Test: {len(test_df)}")
            print(f"{'─'*80}")
            
            X_train = train_df[feature_cols].values
            y_train = train_df['label'].values
            X_val = val_df[feature_cols].values
            y_val = val_df['label'].values
            X_test = test_df[feature_cols].values
            y_test = test_df['label'].values
            
            # Optimizar hiperparámetros
            if optimize and len(X_val) > 0:
                print(f"  Optimizando hiperparámetros...")
                best_params = self.optimize_hyperparameters(
                    X_train, y_train, X_val, y_val, n_trials=n_trials
                )
            else:
                best_params = {
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'n_estimators': 200,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': self.random_state,
                    'objective': 'multi:softmax',
                    'num_class': len(np.unique(y_train))
                }
            
            # Entrenar modelo
            print(f"  Entrenando XGBoost...")
            model = xgb.XGBClassifier(**best_params)

            fit_kwargs = {'verbose': False}
            if len(X_val) > 0:
                fit_kwargs['eval_set'] = [(X_val, y_val)]
                fit_kwargs['early_stopping_rounds'] = 20

            model.fit(X_train, y_train, **fit_kwargs)

            
            # Predicciones
            y_pred_test = model.predict(X_test)
            y_pred_train = model.predict(X_train)
            
            # Métricas
            train_bal_acc = balanced_accuracy_score(y_train, y_pred_train)
            train_f1 = f1_score(y_train, y_pred_train, average='macro')
            test_bal_acc = balanced_accuracy_score(y_test, y_pred_test)
            test_f1 = f1_score(y_test, y_pred_test, average='macro')
            
            fold_results.append({
                'fold': fold + 1,
                'test_subject': test_subject,
                'train_bal_acc': train_bal_acc,
                'train_f1': train_f1,
                'test_bal_acc': test_bal_acc,
                'test_f1': test_f1,
                'n_test': len(X_test)
            })
            
            all_test_true.extend(y_test)
            all_test_pred.extend(y_pred_test)
            
            # Feature importance
            importance = model.feature_importances_
            self.feature_importance[f'fold_{fold+1}'] = dict(zip(feature_cols, importance))
            
            # Guardar fold
            fold_folder = os.path.join(base_output_folder, f'fold_{fold+1:02d}_{test_subject}')
            os.makedirs(fold_folder, exist_ok=True)
            
            model.save_model(os.path.join(fold_folder, 'xgboost_model.json'))
            
            with open(os.path.join(fold_folder, 'config.json'), 'w') as f:
                json.dump({'hyperparameters': best_params, 'metrics': fold_results[-1]}, f, indent=4)
            
            # Matriz de confusión
            cm = confusion_matrix(y_test, y_pred_test)
            self._plot_confusion_matrix(cm, f'Fold {fold+1}',
                                       os.path.join(fold_folder, 'confusion_matrix.png'))
            
            print(f"  ✓ Test Bal Acc: {test_bal_acc:.4f} | F1: {test_f1:.4f}")
        
        # Resultados agregados
        results_df = pd.DataFrame(fold_results)
        summary_folder = os.path.join(base_output_folder, 'summary')
        os.makedirs(summary_folder, exist_ok=True)
        
        results_df.to_csv(os.path.join(summary_folder, 'fold_results.csv'), index=False)
        
        summary = {
            'methodology': 'XGBoost with Nested LOSO',
            'technique': technique,
            'subset': subset,
            'n_folds': len(splits),
            'metrics': {
                'test_bal_acc_mean': results_df['test_bal_acc'].mean(),
                'test_bal_acc_std': results_df['test_bal_acc'].std(),
                'test_f1_mean': results_df['test_f1'].mean(),
                'test_f1_std': results_df['test_f1'].std(),
                'train_bal_acc_mean': results_df['train_bal_acc'].mean(),
                'overfitting_gap': results_df['train_bal_acc'].mean() - results_df['test_bal_acc'].mean()
            }
        }
        
        with open(os.path.join(summary_folder, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)
        
        # Matriz de confusión total
        cm_total = confusion_matrix(all_test_true, all_test_pred)
        self._plot_confusion_matrix(cm_total, 'Total',
                                   os.path.join(summary_folder, 'confusion_matrix_total.png'))
        
        # Imprimir resumen
        print(f"\n{'='*80}")
        print(f"RESUMEN FINAL")
        print(f"{'='*80}")
        print(f"Test Bal Acc: {summary['metrics']['test_bal_acc_mean']*100:.2f}% ± {summary['metrics']['test_bal_acc_std']*100:.2f}%")
        print(f"Test F1:      {summary['metrics']['test_f1_mean']*100:.2f}% ± {summary['metrics']['test_f1_std']*100:.2f}%")
        print(f"{'='*80}\n")
        
        return summary
    
    def _plot_confusion_matrix(self, cm, title, save_path):
        """Plotea matriz de confusión."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Beginner', 'Intermediate', 'Advanced'],
                   yticklabels=['Beginner', 'Intermediate', 'Advanced'])
        plt.title(f'Confusion Matrix - {title}')
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()


# ============================================================================
# PARTE 3: SCRIPT PRINCIPAL
# ============================================================================

def apply_sensor_subset(feature_matrices, subset):
    """Aplica subset de sensores."""
    if subset == 'allStreams':
        return feature_matrices
    elif subset == 'noGforce':
        return np.concatenate((feature_matrices[:, :, :2], feature_matrices[:, :, 18:]), axis=2)
    elif subset == 'noCognionics':
        return np.concatenate((feature_matrices[:, :, :18], feature_matrices[:, :, 22:]), axis=2)
    elif subset == 'noEye':
        return feature_matrices[:, :, 2:]
    elif subset == 'noInsole':
        return np.concatenate((feature_matrices[:, :, :22], feature_matrices[:, :, 58:]), axis=2)
    elif subset == 'noBody':
        return feature_matrices[:, :, :58]
    elif subset == 'onlyGforce':
        return feature_matrices[:, :, 2:18]
    elif subset == 'onlyEye':
        return feature_matrices[:, :, 0:2]
    elif subset == 'onlyInsole':
        return feature_matrices[:, :, 22:58]
    elif subset == 'onlyBody':
        return feature_matrices[:, :, 18:22]
    elif subset == 'onlyCognionics':
        return feature_matrices[:, :, 58:]
    else:
        raise ValueError(f"Subset desconocido: {subset}")


if __name__ == "__main__":
    
    # ========== CONFIGURACIÓN ==========
    DataPath = '/content/drive/My Drive/TFM/data/badminton_data.h5'
    BASE_OUTPUT_FOLDER = '/content/drive/My Drive/TFM/resultados_xgboost/'
    
    SAMPLING_RATE = 100
    N_TRIALS_OPTUNA = 30
    RANDOM_STATE = 42
    
    technique_list = ['forehand', 'backhand']
    sensor_subset_list = ['allStreams', 'noGforce', 'onlyGforce']
    
    # ========== CARGAR DATOS ==========
    print("\n" + "="*80)
    print("PIPELINE XGBOOST CON FEATURE ENGINEERING Y NESTED LOSO")
    print("="*80 + "\n")
    
    with h5py.File(DataPath, 'r') as f:
        feature_matrices = f['example_matrices'][:]
        subject_ids = f['example_subject_ids'][:]
        stroke_types = f['example_label_indexes'][:]
        skill_levels = f['example_skill_level'][:]
    
    subject_ids_str = np.array([x.decode('utf-8') for x in subject_ids])
    
    print(f"✓ Datos cargados: {len(feature_matrices)} muestras")
    
    # ========== ENTRENAR ==========
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    all_results = []
    
    for technique in technique_list:
        indices = np.where(stroke_types == (1 if technique == 'forehand' else 0))[0]
        
        feature_matrices_current = feature_matrices[indices]
        labels_current = skill_levels[indices]
        subject_ids_current = subject_ids_str[indices]
        
        for subset in sensor_subset_list:
            print(f"\n{'='*80}")
            print(f"Procesando: {technique} - {subset}")
            print(f"{'='*80}")
            
            # Aplicar subset
            input_matrices = apply_sensor_subset(feature_matrices_current, subset)
            
            # Feature engineering
            extractor = TimeSeriesFeatureExtractor(sampling_rate=SAMPLING_RATE)
            df_features = extractor.transform(input_matrices, verbose=True)
            
            # Agregar metadata
            df_features['subject_id'] = subject_ids_current
            df_features['label'] = labels_current
            
            # Pipeline XGBoost
            pipeline = XGBoostLOSOPipeline(random_state=RANDOM_STATE)
            
            output_folder = os.path.join(
                BASE_OUTPUT_FOLDER,
                timestamp,
                f'{technique}_{subset}'
            )
            
            try:
                summary = pipeline.train_and_evaluate(
                    df_tabular=df_features,
                    technique=technique,
                    subset=subset,
                    base_output_folder=output_folder,
                    optimize=True,
                    n_trials=N_TRIALS_OPTUNA
                )
                all_results.append(summary)
                print(f"✅ Completado")
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
    
    # ========== RESUMEN FINAL ==========
    if all_results:
        summary_data = []
        for res in all_results:
            summary_data.append({
                'technique': res['technique'],
                'subset': res['subset'],
                'test_bal_acc_mean': res['metrics']['test_bal_acc_mean'] * 100,
                'test_bal_acc_std': res['metrics']['test_bal_acc_std'] * 100,
                'test_f1_mean': res['metrics']['test_f1_mean'] * 100,
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('test_bal_acc_mean', ascending=False)
        
        print(f"\n{'='*80}")
        print("RESUMEN FINAL DE TODOS LOS EXPERIMENTOS")
        print(f"{'='*80}\n")
        print(summary_df.to_string(index=False))
        
        summary_path = os.path.join(BASE_OUTPUT_FOLDER, timestamp, 'all_experiments_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"\n💾 Guardado en: {summary_path}")
