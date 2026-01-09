"""
Pipeline XGBoost con Feature Engineering y Nested LOSO
Versión simplificada con hiperparámetros MANUALES (sin Optuna)
"""

import numpy as np
import pandas as pd
import h5py
import os
from datetime import datetime
from tqdm import tqdm
from scipy import stats
from scipy.fft import fft, fftfreq
import xgboost as xgb
from sklearn.metrics import (balanced_accuracy_score, f1_score, 
                             confusion_matrix, classification_report, accuracy_score)
import matplotlib.pyplot as plt
import seaborn as sns
import json

# ============================================================================
# CONFIGURACIÓN MANUAL DE HIPERPARÁMETROS XGBOOST
# ============================================================================

XGBOOST_PARAMS = {
    'max_depth': 6,                    # Profundidad máxima de árboles (3-10)
    'learning_rate': 0.1,              # Tasa de aprendizaje (0.01-0.3)
    'n_estimators': 200,               # Número de árboles (50-500)
    'subsample': 0.8,                  # Fracción de muestras por árbol (0.6-1.0)
    'colsample_bytree': 0.8,           # Fracción de features por árbol (0.6-1.0)
    'min_child_weight': 3,             # Mínimo peso en nodo hijo (1-10)
    'gamma': 0.1,                      # Regularización: reducción mínima de loss (0-5)
    'reg_alpha': 0.1,                  # Regularización L1 (0-1)
    'reg_lambda': 1.0,                 # Regularización L2 (0-2)
    'random_state': 42,
    'objective': 'multi:softmax',      # Para clasificación multiclase
    'eval_metric': 'mlogloss',
    'early_stopping_rounds': 20        # Parar si no mejora en validación
}

# Configuración general
SAMPLING_RATE = 100                    # Hz
RANDOM_STATE = 42
N_VAL_SUBJECTS = 2                     # Sujetos para validación interna

# ============================================================================
# PARTE 1: FEATURE ENGINEERING
# ============================================================================

class TimeSeriesFeatureExtractor:
    """Extractor de características para secuencias temporales."""
    
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
        
        # Potencia en bandas frecuenciales
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
        
        # Correlaciones entre canales (máximo 10 pares)
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
        
        # Manejar valores infinitos y NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        return df


# ============================================================================
# PARTE 2: PIPELINE XGBOOST CON LOSO
# ============================================================================

class XGBoostLOSOPipeline:
    """Pipeline XGBoost con Leave-One-Subject-Out."""
    
    def __init__(self, xgb_params, random_state=42):
        self.xgb_params = xgb_params
        self.random_state = random_state
        self.feature_importance = {}
    
    def nested_loso_splits(self, df, n_val_subjects_per_class=1):
        """
        Genera splits LOSO con validación interna.
        Selecciona EXACTAMENTE n_val_subjects_per_class de cada clase para validación.
        
        Args:
            df: DataFrame con columnas 'subject_id' y 'label'
            n_val_subjects_per_class: Número de sujetos por clase para validación (default: 1)
        
        Yields:
            (train_df, val_df, test_df, test_subject)
        """
        unique_subjects = df['subject_id'].unique()
        
        for test_subject in unique_subjects:
            # Test: todas las muestras del sujeto
            test_mask = df['subject_id'] == test_subject
            test_df = df[test_mask].copy()
            
            # Sujetos restantes (candidatos para train + val)
            remaining_df = df[~test_mask].copy()
            remaining_subjects = remaining_df['subject_id'].unique()
            
            # Si hay muy pocos sujetos, usar hold-out simple
            if len(remaining_subjects) < 3:
                train_df = remaining_df
                val_df = remaining_df.sample(frac=0.2, random_state=self.random_state)
                yield train_df, val_df, test_df, test_subject
                continue
            
            # Crear mapeo: sujeto -> clase mayoritaria
            subject_to_class = {}
            for subj in remaining_subjects:
                subj_data = remaining_df[remaining_df['subject_id'] == subj]
                majority_class = subj_data['label'].mode()[0]
                subject_to_class[subj] = majority_class
            
            # Agrupar sujetos por clase
            class_to_subjects = {}
            for subj, cls in subject_to_class.items():
                if cls not in class_to_subjects:
                    class_to_subjects[cls] = []
                class_to_subjects[cls].append(subj)
            
            # Seleccionar exactamente n_val_subjects_per_class de cada clase
            val_subjects = []
            
            for cls in sorted(class_to_subjects.keys()):
                subjects_in_class = class_to_subjects[cls]
                
                # Si hay suficientes sujetos en esta clase
                if len(subjects_in_class) >= n_val_subjects_per_class + 1:
                    # Seleccionar n_val_subjects_per_class aleatoriamente
                    selected = np.random.choice(
                        subjects_in_class, 
                        size=n_val_subjects_per_class, 
                        replace=False
                    )
                    val_subjects.extend(selected)
                elif len(subjects_in_class) > 0:
                    # Si hay pocos sujetos, tomar solo 1 para validación
                    selected = np.random.choice(subjects_in_class, size=1, replace=False)
                    val_subjects.extend(selected)
            
            # Si por alguna razón no hay sujetos de validación, usar hold-out
            if len(val_subjects) == 0:
                train_df = remaining_df
                val_df = remaining_df.sample(frac=0.2, random_state=self.random_state)
            else:
                val_df = remaining_df[remaining_df['subject_id'].isin(val_subjects)].copy()
                train_df = remaining_df[~remaining_df['subject_id'].isin(val_subjects)].copy()
            
            yield train_df, val_df, test_df, test_subject
    
    def train_and_evaluate(self, df_tabular, technique, subset, base_output_folder):
        """Pipeline completo de entrenamiento y evaluación."""
        
        print(f"\n{'='*80}")
        print(f"XGBoost con Nested LOSO")
        print(f"Técnica: {technique} | Subset: {subset}")
        print(f"{'='*80}\n")
        
        print(f"📋 Hiperparámetros XGBoost:")
        for key, value in self.xgb_params.items():
            if key != 'num_class':
                print(f"   {key}: {value}")
        print()
        
        os.makedirs(base_output_folder, exist_ok=True)
        
        # Separar features de metadata
        feature_cols = [col for col in df_tabular.columns 
                       if col not in ['subject_id', 'label']]
        
        fold_results = []
        all_test_true = []
        all_test_pred = []
        
        # Generar splits LOSO
        splits = list(self.nested_loso_splits(df_tabular, n_val_subjects=N_VAL_SUBJECTS))
        
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
        
        # Iterar sobre folds
        for fold, (train_df, val_df, test_df, test_subject) in enumerate(splits):
            print(f"\n{'─'*80}")
            print(f"LOSO Fold {fold+1}/{len(splits)} - Test Subject: {test_subject}")
            print(f"  Train: {len(train_df)} samples from {len(train_df['subject_id'].unique())} subjects")
            print(f"  Val:   {len(val_df)} samples from {len(val_df['subject_id'].unique())} subjects")
            print(f"  Test:  {len(test_df)} samples")
            print(f"{'─'*80}")
            
            # Preparar datos
            X_train = train_df[feature_cols].values
            y_train = train_df['label'].values
            X_val = val_df[feature_cols].values
            y_val = val_df['label'].values
            X_test = test_df[feature_cols].values
            y_test = test_df['label'].values
            
            # Configurar parámetros (añadir num_class)
            params = self.xgb_params.copy()
            params['num_class'] = len(np.unique(y_train))
            
            # Entrenar modelo
            print(f"  Entrenando XGBoost...")
            model = xgb.XGBClassifier(**params)
            
            # Entrenar con early stopping si hay validación
            if len(X_val) > 0:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            else:
                model.fit(X_train, y_train, verbose=False)
            
            # Predicciones
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Métricas
            train_acc = accuracy_score(y_train, y_pred_train)
            train_bal_acc = balanced_accuracy_score(y_train, y_pred_train)
            train_f1 = f1_score(y_train, y_pred_train, average='macro')
            
            test_acc = accuracy_score(y_test, y_pred_test)
            test_bal_acc = balanced_accuracy_score(y_test, y_pred_test)
            test_f1 = f1_score(y_test, y_pred_test, average='macro')
            
            fold_results.append({
                'fold': fold + 1,
                'test_subject': test_subject,
                'train_acc': train_acc,
                'train_bal_acc': train_bal_acc,
                'train_f1': train_f1,
                'test_acc': test_acc,
                'test_bal_acc': test_bal_acc,
                'test_f1': test_f1,
                'n_train': len(X_train),
                'n_val': len(X_val),
                'n_test': len(X_test)
            })
            
            # Guardar predicciones
            all_test_true.extend(y_test)
            all_test_pred.extend(y_pred_test)
            
            # Feature importance
            importance = model.feature_importances_
            self.feature_importance[f'fold_{fold+1}'] = dict(zip(feature_cols, importance))
            
            # Crear carpeta del fold
            fold_folder = os.path.join(base_output_folder, f'fold_{fold+1:02d}_{test_subject}')
            os.makedirs(fold_folder, exist_ok=True)
            
            # Guardar modelo
            model.save_model(os.path.join(fold_folder, 'xgboost_model.json'))
            
            # Guardar configuración y métricas
            fold_config = {
                'fold': fold + 1,
                'test_subject': test_subject,
                'hyperparameters': params,
                'metrics': fold_results[-1]
            }
            with open(os.path.join(fold_folder, 'config.json'), 'w') as f:
                json.dump(fold_config, f, indent=4)
            
            # Matriz de confusión del fold
            cm = confusion_matrix(y_test, y_pred_test)
            self._plot_confusion_matrix(
                cm, 
                f'Fold {fold+1} - Test Subject: {test_subject}',
                os.path.join(fold_folder, 'confusion_matrix.png')
            )
            
            # Feature importance del fold (top 20)
            self._plot_feature_importance(
                importance, 
                feature_cols,
                f'Feature Importance - Fold {fold+1}',
                os.path.join(fold_folder, 'feature_importance.png'),
                top_n=20
            )
            
            print(f"  ✓ Train Bal Acc: {train_bal_acc:.4f} | Test Bal Acc: {test_bal_acc:.4f} | F1: {test_f1:.4f}")
        
        # ========== RESULTADOS AGREGADOS ==========
        
        results_df = pd.DataFrame(fold_results)
        summary_folder = os.path.join(base_output_folder, 'summary')
        os.makedirs(summary_folder, exist_ok=True)
        
        # Guardar resultados por fold
        results_df.to_csv(os.path.join(summary_folder, 'fold_results.csv'), index=False)
        
        # Métricas promedio
        summary = {
            'methodology': 'XGBoost with Nested Leave-One-Subject-Out',
            'technique': technique,
            'subset': subset,
            'n_folds': len(splits),
            'hyperparameters': self.xgb_params,
            'metrics': {
                'train_acc_mean': results_df['train_acc'].mean(),
                'train_acc_std': results_df['train_acc'].std(),
                'train_bal_acc_mean': results_df['train_bal_acc'].mean(),
                'train_bal_acc_std': results_df['train_bal_acc'].std(),
                'test_acc_mean': results_df['test_acc'].mean(),
                'test_acc_std': results_df['test_acc'].std(),
                'test_bal_acc_mean': results_df['test_bal_acc'].mean(),
                'test_bal_acc_std': results_df['test_bal_acc'].std(),
                'train_f1_mean': results_df['train_f1'].mean(),
                'train_f1_std': results_df['train_f1'].std(),
                'test_f1_mean': results_df['test_f1'].mean(),
                'test_f1_std': results_df['test_f1'].std(),
                'overfitting_gap': results_df['train_bal_acc'].mean() - results_df['test_bal_acc'].mean()
            }
        }
        
        # Guardar resumen
        with open(os.path.join(summary_folder, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)
        
        # Matriz de confusión agregada
        cm_total = confusion_matrix(all_test_true, all_test_pred)
        self._plot_confusion_matrix(
            cm_total,
            'Confusion Matrix - All Folds Aggregated',
            os.path.join(summary_folder, 'confusion_matrix_total.png')
        )
        
        # Classification report
        report = classification_report(
            all_test_true, all_test_pred,
            target_names=['Beginner', 'Intermediate', 'Advanced'],
            output_dict=True
        )
        with open(os.path.join(summary_folder, 'classification_report.json'), 'w') as f:
            json.dump(report, f, indent=4)
        
        # Feature importance promedio
        avg_importance = self._average_feature_importance(feature_cols)
        self._plot_feature_importance(
            list(avg_importance.values()),
            list(avg_importance.keys()),
            'Average Feature Importance Across All Folds',
            os.path.join(summary_folder, 'feature_importance_avg.png'),
            top_n=30
        )
        
        # Gráficos de resultados por fold
        self._plot_fold_results(results_df, summary_folder)
        
        # Imprimir resumen
        print(f"\n{'='*80}")
        print(f"RESUMEN FINAL - XGBoost LOSO")
        print(f"{'='*80}")
        print(f"Técnica: {technique} | Subset: {subset}")
        print(f"Número de folds: {len(splits)}")
        print(f"")
        print(f"📊 Métricas (media ± std):")
        print(f"  Test Accuracy:          {summary['metrics']['test_acc_mean']*100:.2f}% ± {summary['metrics']['test_acc_std']*100:.2f}%")
        print(f"  Test Balanced Accuracy: {summary['metrics']['test_bal_acc_mean']*100:.2f}% ± {summary['metrics']['test_bal_acc_std']*100:.2f}%")
        print(f"  Test Macro F1:          {summary['metrics']['test_f1_mean']*100:.2f}% ± {summary['metrics']['test_f1_std']*100:.2f}%")
        print(f"  Train Balanced Accuracy: {summary['metrics']['train_bal_acc_mean']*100:.2f}% ± {summary['metrics']['train_bal_acc_std']*100:.2f}%")
        print(f"  Overfitting gap:         {summary['metrics']['overfitting_gap']*100:.2f}%")
        print(f"")
        print(f"📁 Resultados guardados en: {base_output_folder}")
        print(f"{'='*80}\n")
        
        return summary
    
    def _average_feature_importance(self, feature_names):
        """Calcula importancia promedio de features."""
        avg_importance = {name: 0 for name in feature_names}
        
        for fold_imp in self.feature_importance.values():
            for name, imp in fold_imp.items():
                avg_importance[name] += imp
        
        n_folds = len(self.feature_importance)
        avg_importance = {k: v/n_folds for k, v in avg_importance.items()}
        
        # Ordenar por importancia
        avg_importance = dict(sorted(avg_importance.items(), 
                                    key=lambda x: x[1], reverse=True))
        
        return avg_importance
    
    def _plot_confusion_matrix(self, cm, title, save_path):
        """Plotea matriz de confusión."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Beginner', 'Intermediate', 'Advanced'],
                   yticklabels=['Beginner', 'Intermediate', 'Advanced'])
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
    
    def _plot_feature_importance(self, importance, feature_names, title, save_path, top_n=20):
        """Plotea feature importance."""
        indices = np.argsort(importance)[-top_n:]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), [importance[i] for i in indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
    
    def _plot_fold_results(self, results_df, save_folder):
        """Plotea resultados por fold."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        x = range(1, len(results_df) + 1)
        
        # Test Balanced Accuracy
        axes[0, 0].bar(x, results_df['test_bal_acc'] * 100, alpha=0.7, color='steelblue')
        axes[0, 0].axhline(results_df['test_bal_acc'].mean() * 100, 
                          color='red', linestyle='--', 
                          label=f'Mean: {results_df["test_bal_acc"].mean()*100:.2f}%')
        axes[0, 0].set_xlabel('Fold')
        axes[0, 0].set_ylabel('Test Balanced Accuracy (%)')
        axes[0, 0].set_title('Test Balanced Accuracy by Fold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Test F1
        axes[0, 1].bar(x, results_df['test_f1'] * 100, alpha=0.7, color='forestgreen')
        axes[0, 1].axhline(results_df['test_f1'].mean() * 100, 
                          color='red', linestyle='--',
                          label=f'Mean: {results_df["test_f1"].mean()*100:.2f}%')
        axes[0, 1].set_xlabel('Fold')
        axes[0, 1].set_ylabel('Test Macro F1 (%)')
        axes[0, 1].set_title('Test Macro F1 by Fold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Train vs Test
        width = 0.35
        x_pos = np.arange(len(results_df))
        axes[1, 0].bar(x_pos - width/2, results_df['train_bal_acc'] * 100, 
                      width, label='Train', alpha=0.7, color='skyblue')
        axes[1, 0].bar(x_pos + width/2, results_df['test_bal_acc'] * 100, 
                      width, label='Test', alpha=0.7, color='coral')
        axes[1, 0].set_xlabel('Fold')
        axes[1, 0].set_ylabel('Balanced Accuracy (%)')
        axes[1, 0].set_title('Train vs Test Balanced Accuracy')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels([f'{i+1}' for i in range(len(results_df))])
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Sample sizes
        axes[1, 1].bar(x, results_df['n_test'], alpha=0.7, color='mediumpurple')
        axes[1, 1].set_xlabel('Fold')
        axes[1, 1].set_ylabel('Number of Test Samples')
        axes[1, 1].set_title('Test Set Size by Fold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, 'fold_results_plots.png'), dpi=150)
        plt.close()


# ============================================================================
# PARTE 3: UTILIDADES
# ============================================================================

def apply_sensor_subset(feature_matrices, subset):
    """Aplica subset de sensores a las matrices."""
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


# ============================================================================
# SCRIPT PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    
    # ========== RUTAS (AJUSTAR SEGÚN TU CONFIGURACIÓN) ==========
    DataPath = '/content/drive/My Drive/TFM/data/badminton_data.h5'
    BASE_OUTPUT_FOLDER = '/content/drive/My Drive/TFM/resultados_xgboost_manual/'
    
    # ========== CONFIGURACIONES A PROBAR ==========
    technique_list = ['forehand', 'backhand']
    sensor_subset_list = ['allStreams', 'noGforce', 'onlyGforce']
    
    # ========== CARGAR DATOS ==========
    print("\n" + "="*80)
    print("PIPELINE XGBOOST CON HIPERPARÁMETROS MANUALES")
    print("="*80 + "\n")
    
    print("📂 Cargando datos...")
    with h5py.File(DataPath, 'r') as f:
        feature_matrices = f['example_matrices'][:]
        subject_ids = f['example_subject_ids'][:]
        stroke_types = f['example_label_indexes'][:]
        skill_levels = f['example_skill_level'][:]
    
    subject_ids_str = np.array([x.decode('utf-8') for x in subject_ids])
    
    print(f"✓ Datos cargados:")
    print(f"  - Muestras totales: {len(feature_matrices)}")
    print(f"  - Shape: {feature_matrices.shape}")
    print(f"  - Sujetos únicos: {len(np.unique(subject_ids_str))}")
    
    # ========== ENTRENAR PARA CADA CONFIGURACIÓN ==========
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    all_results = []
    
    for technique in technique_list:
        
        # Filtrar por técnica
        print(f"\n{'='*80}")
        print(f"Procesando técnica: {technique.upper()}")
        print(f"{'='*80}")
        
        if technique == 'forehand':
            indices = np.where(stroke_types == 1)[0]
        elif technique == 'backhand':
            indices = np.where(stroke_types == 0)[0]
        
        feature_matrices_current = feature_matrices[indices]
        labels_current = skill_levels[indices]
        subject_ids_current = subject_ids_str[indices]
        
        print(f"✓ Datos filtrados: {len(feature_matrices_current)} muestras")
        
        # Probar cada subset
        for subset in sensor_subset_list:
            
            print(f"\n{'─'*80}")
            print(f"Subset: {subset}")
            print(f"{'─'*80}")
            
            # Aplicar subset de sensores
            input_matrices = apply_sensor_subset(feature_matrices_current, subset)
            print(f"✓ Shape después de subset: {input_matrices.shape}")
            
            # ========== FEATURE ENGINEERING ==========
            extractor = TimeSeriesFeatureExtractor(sampling_rate=SAMPLING_RATE)
            df_features = extractor.transform(input_matrices, verbose=True)
            
            # Agregar metadata
            df_features['subject_id'] = subject_ids_current
            df_features['label'] = labels_current
            
            print(f"✓ Features extraídas: {len(extractor.feature_names)} características")
            
            # ========== ENTRENAR CON XGBOOST ==========
            pipeline = XGBoostLOSOPipeline(
                xgb_params=XGBOOST_PARAMS,
                random_state=RANDOM_STATE
            )
            
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
                    base_output_folder=output_folder
                )
                
                all_results.append(summary)
                print(f"✅ Completado: {technique} - {subset}")
                
            except Exception as e:
                print(f"❌ Error en {technique} - {subset}: {str(e)}")
                import traceback
                traceback.print_exc()
    
    # ========== RESUMEN FINAL DE TODOS LOS EXPERIMENTOS ==========
    
    if all_results:
        print(f"\n{'#'*80}")
        print(f"# RESUMEN FINAL DE TODOS LOS EXPERIMENTOS")
        print(f"{'#'*80}\n")
        
        summary_data = []
        for res in all_results:
            summary_data.append({
                'technique': res['technique'],
                'subset': res['subset'],
                'n_folds': res['n_folds'],
                'test_acc_mean': res['metrics']['test_acc_mean'] * 100,
                'test_acc_std': res['metrics']['test_acc_std'] * 100,
                'test_bal_acc_mean': res['metrics']['test_bal_acc_mean'] * 100,
                'test_bal_acc_std': res['metrics']['test_bal_acc_std'] * 100,
                'test_f1_mean': res['metrics']['test_f1_mean'] * 100,
                'test_f1_std': res['metrics']['test_f1_std'] * 100,
                'overfitting_gap': res['metrics']['overfitting_gap'] * 100
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('test_bal_acc_mean', ascending=False)
        
        print("📊 Resultados ordenados por Test Balanced Accuracy:\n")
        print(summary_df.to_string(index=False))
        
        # Guardar resumen
        summary_path = os.path.join(BASE_OUTPUT_FOLDER, timestamp, 'all_experiments_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        
        print(f"\n\n💾 Resumen guardado en: {summary_path}")
        
        # Estadísticas por técnica
        print(f"\n{'─'*80}")
        print("📈 Estadísticas por técnica:")
        print(f"{'─'*80}\n")
        
        for tech in technique_list:
            tech_data = summary_df[summary_df['technique'] == tech]
            if len(tech_data) > 0:
                print(f"{tech.upper()}:")
                print(f"  Mejor Bal Acc:  {tech_data['test_bal_acc_mean'].max():.2f}%")
                print(f"  Media Bal Acc:  {tech_data['test_bal_acc_mean'].mean():.2f}%")
                print(f"  Mejor subset:   {tech_data.iloc[0]['subset']}")
                print()
    
    else:
        print("\n⚠️  No se completaron experimentos exitosamente")
    
    print(f"\n{'#'*80}")
    print(f"# PIPELINE COMPLETADO")
    print(f"# Resultados en: {os.path.join(BASE_OUTPUT_FOLDER, timestamp)}")
    print(f"{'#'*80}\n")