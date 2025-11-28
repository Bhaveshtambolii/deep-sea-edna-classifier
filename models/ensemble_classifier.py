"""
Ensemble Methods for eDNA Classification
Train multiple models and combine predictions for improved accuracy
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import List, Dict, Tuple, Optional, Union
import json
from pathlib import Path
from collections import Counter
import pickle

from improved_cnn_classifier import (
    ImprovedCNNClassifier, 
    create_improved_model,
    ResidualBlock,
    SelfAttention,
    ChannelAttention,
    MultiScaleConv
)


class EnsembleClassifier:
    """
    Ensemble of multiple CNN classifiers with various combination strategies
    """
    
    def __init__(
        self,
        input_length: int,
        num_classes: int,
        encoding_dim: int = 5,
        n_models: int = 5,
        architectures: Optional[List[str]] = None
    ):
        """
        Initialize ensemble
        
        Args:
            input_length: Sequence length
            num_classes: Number of classes
            encoding_dim: Encoding dimension (5 for one-hot DNA)
            n_models: Number of models in ensemble
            architectures: List of architectures to use
        """
        self.input_length = input_length
        self.num_classes = num_classes
        self.encoding_dim = encoding_dim
        self.n_models = n_models
        
        # Default: mix of architectures
        if architectures is None:
            architectures = ['attention_resnet', 'multiscale', 'attention_resnet',
                           'deep_resnet', 'attention_resnet'][:n_models]
        self.architectures = architectures
        
        self.models: List[ImprovedCNNClassifier] = []
        self.weights: np.ndarray = None
        self.trained = False
        
    def create_models(self, seeds: Optional[List[int]] = None):
        """
        Create ensemble models with different random seeds
        
        Args:
            seeds: Random seeds for each model
        """
        if seeds is None:
            seeds = [42 + i * 17 for i in range(self.n_models)]
        
        self.models = []
        for i in range(self.n_models):
            # Set seed for reproducibility
            tf.random.set_seed(seeds[i])
            np.random.seed(seeds[i])
            
            # Create model with architecture rotation
            arch = self.architectures[i % len(self.architectures)]
            
            model = create_improved_model(
                input_length=self.input_length,
                num_classes=self.num_classes,
                encoding_dim=self.encoding_dim,
                architecture=arch
            )
            self.models.append(model)
            
        print(f"Created {self.n_models} models:")
        for i, (model, arch) in enumerate(zip(self.models, self.architectures)):
            print(f"  Model {i+1}: {arch}")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        use_class_weights: bool = True,
        verbose: int = 1,
        bootstrap: bool = True,
        bootstrap_ratio: float = 0.8
    ) -> List[Dict]:
        """
        Train all models in ensemble
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs: Training epochs per model
            batch_size: Batch size
            use_class_weights: Use balanced class weights
            verbose: Verbosity level
            bootstrap: Use bootstrap sampling for each model
            bootstrap_ratio: Fraction of data for bootstrap samples
            
        Returns:
            Training histories for each model
        """
        if not self.models:
            self.create_models()
        
        histories = []
        val_accuracies = []
        
        for i, model in enumerate(self.models):
            print(f"\n{'='*60}")
            print(f"Training Model {i+1}/{self.n_models} ({self.architectures[i]})")
            print('='*60)
            
            # Bootstrap sampling
            if bootstrap:
                n_samples = int(len(X_train) * bootstrap_ratio)
                indices = np.random.choice(len(X_train), n_samples, replace=True)
                X_train_boot = X_train[indices]
                y_train_boot = y_train[indices]
            else:
                X_train_boot = X_train
                y_train_boot = y_train
            
            # Train model
            history = model.train(
                X_train_boot, y_train_boot,
                X_val, y_val,
                epochs=epochs,
                batch_size=batch_size,
                use_class_weights=use_class_weights,
                verbose=verbose
            )
            
            # Evaluate
            metrics = model.evaluate(X_val, y_val)
            val_accuracies.append(metrics['accuracy'])
            
            histories.append({
                'model_idx': i,
                'architecture': self.architectures[i],
                'val_accuracy': float(metrics['accuracy']),
                'val_top3_accuracy': float(metrics.get('top3_accuracy', 0)),
                'val_loss': float(metrics['loss'])
            })
            
            print(f"Model {i+1} Val Accuracy: {metrics['accuracy']:.4f}")
        
        # Compute model weights based on validation performance
        self._compute_weights(val_accuracies)
        self.trained = True
        
        return histories
    
    def _compute_weights(self, accuracies: List[float], method: str = 'softmax'):
        """
        Compute ensemble weights based on validation accuracy
        
        Args:
            accuracies: List of validation accuracies
            method: Weighting method ('equal', 'linear', 'softmax')
        """
        if method == 'equal':
            self.weights = np.ones(self.n_models) / self.n_models
        elif method == 'linear':
            acc_array = np.array(accuracies)
            self.weights = acc_array / acc_array.sum()
        elif method == 'softmax':
            acc_array = np.array(accuracies)
            # Temperature scaling
            temperature = 0.1
            exp_acc = np.exp((acc_array - acc_array.max()) / temperature)
            self.weights = exp_acc / exp_acc.sum()
        
        print(f"\nEnsemble weights ({method}):")
        for i, w in enumerate(self.weights):
            print(f"  Model {i+1}: {w:.4f}")
    
    def predict_proba(
        self, 
        X: np.ndarray, 
        batch_size: int = 32,
        combination: str = 'weighted_average'
    ) -> np.ndarray:
        """
        Get probability predictions from ensemble
        
        Args:
            X: Input data
            batch_size: Batch size
            combination: How to combine predictions
                'average': Simple average
                'weighted_average': Weighted by validation accuracy
                'voting': Soft voting
                
        Returns:
            Combined probability predictions
        """
        if not self.trained:
            raise ValueError("Ensemble not trained yet")
        
        # Get predictions from all models
        all_preds = np.stack([
            model.predict(X, batch_size=batch_size)
            for model in self.models
        ])
        
        if combination == 'average':
            return np.mean(all_preds, axis=0)
        
        elif combination == 'weighted_average':
            # Weighted combination
            weighted = np.zeros_like(all_preds[0])
            for i, pred in enumerate(all_preds):
                weighted += self.weights[i] * pred
            return weighted
        
        elif combination == 'voting':
            # Soft voting with normalization
            return np.mean(all_preds, axis=0)
        
        elif combination == 'max':
            # Take max probability across models
            return np.max(all_preds, axis=0)
        
        else:
            return np.mean(all_preds, axis=0)
    
    def predict(
        self, 
        X: np.ndarray, 
        batch_size: int = 32,
        combination: str = 'weighted_average'
    ) -> np.ndarray:
        """
        Get class predictions from ensemble
        
        Args:
            X: Input data
            batch_size: Batch size
            combination: Combination method
            
        Returns:
            Class predictions
        """
        proba = self.predict_proba(X, batch_size, combination)
        return np.argmax(proba, axis=1)
    
    def predict_with_confidence(
        self, 
        X: np.ndarray, 
        batch_size: int = 32
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get predictions with confidence metrics
        
        Args:
            X: Input data
            batch_size: Batch size
            
        Returns:
            predictions, confidence_scores, agreement_scores
        """
        # Get individual model predictions
        all_preds = np.stack([
            model.predict(X, batch_size=batch_size)
            for model in self.models
        ])
        
        # Combined prediction
        combined = np.zeros_like(all_preds[0])
        for i, pred in enumerate(all_preds):
            combined += self.weights[i] * pred
        
        predictions = np.argmax(combined, axis=1)
        confidence = np.max(combined, axis=1)
        
        # Model agreement (how many models agree on prediction)
        individual_preds = np.argmax(all_preds, axis=2)  # [n_models, n_samples]
        agreement = np.array([
            np.mean(individual_preds[:, i] == predictions[i])
            for i in range(len(X))
        ])
        
        return predictions, confidence, agreement
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32
    ) -> Dict:
        """
        Evaluate ensemble performance
        
        Args:
            X: Test data
            y: Test labels
            batch_size: Batch size
            
        Returns:
            Evaluation metrics
        """
        # Get predictions
        proba = self.predict_proba(X, batch_size)
        y_pred = np.argmax(proba, axis=1)
        y_true = np.argmax(y, axis=1) if len(y.shape) > 1 else y
        
        # Accuracy
        accuracy = np.mean(y_pred == y_true)
        
        # Top-3 accuracy
        top3_preds = np.argsort(proba, axis=1)[:, -3:]
        top3_correct = np.array([y_true[i] in top3_preds[i] for i in range(len(y_true))])
        top3_accuracy = np.mean(top3_correct)
        
        # Top-5 accuracy
        top5_preds = np.argsort(proba, axis=1)[:, -5:]
        top5_correct = np.array([y_true[i] in top5_preds[i] for i in range(len(y_true))])
        top5_accuracy = np.mean(top5_correct)
        
        # Individual model accuracies
        individual_accuracies = []
        for model in self.models:
            metrics = model.evaluate(X, y, batch_size)
            individual_accuracies.append(metrics['accuracy'])
        
        return {
            'ensemble_accuracy': float(accuracy),
            'ensemble_top3_accuracy': float(top3_accuracy),
            'ensemble_top5_accuracy': float(top5_accuracy),
            'individual_accuracies': individual_accuracies,
            'mean_individual_accuracy': float(np.mean(individual_accuracies)),
            'ensemble_gain': float(accuracy - np.mean(individual_accuracies)),
            'predictions': proba,
            'y_pred': y_pred,
            'y_true': y_true
        }
    
    def save(self, output_dir: str):
        """Save ensemble to directory"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save each model
        for i, model in enumerate(self.models):
            model_dir = output_path / f'model_{i}'
            model_dir.mkdir(exist_ok=True)
            model.save(str(model_dir / 'model.h5'))
        
        # Save config and weights
        config = {
            'input_length': self.input_length,
            'num_classes': self.num_classes,
            'encoding_dim': self.encoding_dim,
            'n_models': self.n_models,
            'architectures': self.architectures,
            'weights': self.weights.tolist() if self.weights is not None else None
        }
        
        with open(output_path / 'ensemble_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Ensemble saved to {output_dir}")
    
    @classmethod
    def load(cls, input_dir: str) -> 'EnsembleClassifier':
        """Load ensemble from directory"""
        input_path = Path(input_dir)
        
        # Load config
        with open(input_path / 'ensemble_config.json', 'r') as f:
            config = json.load(f)
        
        ensemble = cls(
            input_length=config['input_length'],
            num_classes=config['num_classes'],
            encoding_dim=config['encoding_dim'],
            n_models=config['n_models'],
            architectures=config['architectures']
        )
        
        # Load models
        ensemble.models = []
        for i in range(config['n_models']):
            model_path = str(input_path / f'model_{i}' / 'model.h5')
            model = ImprovedCNNClassifier.load(model_path)
            ensemble.models.append(model)
        
        if config['weights']:
            ensemble.weights = np.array(config['weights'])
        
        ensemble.trained = True
        return ensemble


class StackingEnsemble:
    """
    Stacking ensemble with meta-learner
    """
    
    def __init__(
        self,
        base_models: List[ImprovedCNNClassifier],
        num_classes: int
    ):
        """
        Initialize stacking ensemble
        
        Args:
            base_models: List of trained base classifiers
            num_classes: Number of classes
        """
        self.base_models = base_models
        self.num_classes = num_classes
        self.meta_model = None
    
    def _create_meta_features(
        self, 
        X: np.ndarray, 
        batch_size: int = 32
    ) -> np.ndarray:
        """Create meta-features from base model predictions"""
        features = []
        for model in self.base_models:
            preds = model.predict(X, batch_size=batch_size)
            features.append(preds)
        return np.hstack(features)
    
    def train_meta_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32
    ):
        """
        Train meta-learner on base model predictions
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs: Training epochs
            batch_size: Batch size
        """
        print("Creating meta-features...")
        meta_train = self._create_meta_features(X_train, batch_size)
        meta_val = self._create_meta_features(X_val, batch_size)
        
        print(f"Meta-feature shape: {meta_train.shape}")
        
        # Create simple meta-model
        input_dim = meta_train.shape[1]
        
        self.meta_model = keras.Sequential([
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.meta_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Training meta-model...")
        self.meta_model.fit(
            meta_train, y_train,
            validation_data=(meta_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
            ],
            verbose=1
        )
    
    def predict(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """Get predictions from stacking ensemble"""
        meta_features = self._create_meta_features(X, batch_size)
        return self.meta_model.predict(meta_features)


class BaggingClassifier:
    """
    Bagging ensemble with random subsets
    """
    
    def __init__(
        self,
        input_length: int,
        num_classes: int,
        encoding_dim: int = 5,
        n_estimators: int = 10,
        max_samples: float = 0.8
    ):
        self.input_length = input_length
        self.num_classes = num_classes
        self.encoding_dim = encoding_dim
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.models = []
        
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 30,
        batch_size: int = 32
    ):
        """Train bagging ensemble"""
        self.models = []
        
        for i in range(self.n_estimators):
            print(f"\nTraining estimator {i+1}/{self.n_estimators}")
            
            # Bootstrap sample
            n_samples = int(len(X_train) * self.max_samples)
            indices = np.random.choice(len(X_train), n_samples, replace=True)
            
            # Create and train model
            model = create_improved_model(
                input_length=self.input_length,
                num_classes=self.num_classes,
                encoding_dim=self.encoding_dim,
                architecture='attention_resnet'
            )
            
            model.train(
                X_train[indices], y_train[indices],
                X_val, y_val,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0
            )
            
            metrics = model.evaluate(X_val, y_val)
            print(f"  Val accuracy: {metrics['accuracy']:.4f}")
            
            self.models.append(model)
    
    def predict(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """Average predictions from all estimators"""
        all_preds = np.stack([
            model.predict(X, batch_size=batch_size)
            for model in self.models
        ])
        return np.mean(all_preds, axis=0)


def train_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    input_length: int,
    num_classes: int,
    output_dir: str,
    n_models: int = 5,
    epochs: int = 50,
    batch_size: int = 32
) -> Dict:
    """
    Convenience function to train and evaluate ensemble
    
    Returns:
        Dictionary with results
    """
    print("="*60)
    print("ENSEMBLE TRAINING")
    print("="*60)
    
    # Create ensemble
    ensemble = EnsembleClassifier(
        input_length=input_length,
        num_classes=num_classes,
        n_models=n_models
    )
    
    # Train
    histories = ensemble.train(
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        bootstrap=True
    )
    
    # Evaluate
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    
    metrics = ensemble.evaluate(X_test, y_test, batch_size)
    
    print(f"\nEnsemble Accuracy: {metrics['ensemble_accuracy']:.4f}")
    print(f"Ensemble Top-3 Accuracy: {metrics['ensemble_top3_accuracy']:.4f}")
    print(f"Mean Individual Accuracy: {metrics['mean_individual_accuracy']:.4f}")
    print(f"Ensemble Gain: +{metrics['ensemble_gain']*100:.2f}%")
    
    # Save
    ensemble.save(output_dir)
    
    # Save results
    results = {
        'ensemble_accuracy': metrics['ensemble_accuracy'],
        'ensemble_top3_accuracy': metrics['ensemble_top3_accuracy'],
        'individual_accuracies': metrics['individual_accuracies'],
        'ensemble_gain': metrics['ensemble_gain'],
        'n_models': n_models,
        'architectures': ensemble.architectures
    }
    
    with open(Path(output_dir) / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == '__main__':
    print("Ensemble Classification Module")
    print("="*60)
    print("Available ensemble methods:")
    print("  1. EnsembleClassifier: Weighted averaging of multiple models")
    print("  2. StackingEnsemble: Meta-learner on base model outputs")
    print("  3. BaggingClassifier: Bootstrap aggregating")
    print()
    print("Usage:")
    print("  from ensemble_classifier import EnsembleClassifier")
    print("  ensemble = EnsembleClassifier(input_length=500, num_classes=126, n_models=5)")
    print("  ensemble.train(X_train, y_train, X_val, y_val)")
    print("  predictions = ensemble.predict(X_test)")
