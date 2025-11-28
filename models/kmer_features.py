"""
K-mer Feature Extraction and Hybrid CNN Model
Combines sequence-level CNN with k-mer frequency features
"""

import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Optional
import itertools
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.preprocessing import StandardScaler
import json
from pathlib import Path

from improved_cnn_classifier import (
    ResidualBlock, 
    ChannelAttention, 
    SelfAttention,
    ImprovedCNNClassifier
)


class KmerExtractor:
    """
    Extract k-mer frequency features from DNA sequences
    """
    
    def __init__(
        self, 
        k: int = 4, 
        alphabet: str = 'ACGT',
        normalize: bool = True
    ):
        """
        Initialize k-mer extractor
        
        Args:
            k: K-mer size (typically 3-6)
            alphabet: DNA alphabet
            normalize: Normalize frequencies to sum to 1
        """
        self.k = k
        self.alphabet = alphabet
        self.normalize = normalize
        
        # Generate all possible k-mers
        self.kmers = [''.join(p) for p in itertools.product(alphabet, repeat=k)]
        self.kmer_to_idx = {kmer: i for i, kmer in enumerate(self.kmers)}
        self.n_features = len(self.kmers)
        
        # Scaler for standardization
        self.scaler = StandardScaler()
        self.fitted = False
        
    def extract(self, sequence: str) -> np.ndarray:
        """
        Extract k-mer frequencies from single sequence
        
        Args:
            sequence: DNA sequence string
            
        Returns:
            K-mer frequency vector
        """
        # Count k-mers
        counts = Counter()
        seq = sequence.upper()
        
        for i in range(len(seq) - self.k + 1):
            kmer = seq[i:i + self.k]
            if kmer in self.kmer_to_idx:
                counts[kmer] += 1
        
        # Convert to frequency vector
        freq = np.zeros(self.n_features, dtype=np.float32)
        total = sum(counts.values())
        
        if total > 0:
            for kmer, count in counts.items():
                idx = self.kmer_to_idx[kmer]
                freq[idx] = count / total if self.normalize else count
        
        return freq
    
    def extract_batch(self, sequences: List[str]) -> np.ndarray:
        """
        Extract k-mer features from multiple sequences
        
        Args:
            sequences: List of DNA sequences
            
        Returns:
            Feature matrix [n_sequences, n_features]
        """
        features = np.array([self.extract(seq) for seq in sequences])
        return features
    
    def fit_scaler(self, features: np.ndarray):
        """Fit standardization scaler on training data"""
        self.scaler.fit(features)
        self.fitted = True
        
    def transform(self, features: np.ndarray) -> np.ndarray:
        """Apply standardization"""
        if not self.fitted:
            raise ValueError("Scaler not fitted. Call fit_scaler first.")
        return self.scaler.transform(features)
    
    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Fit scaler and transform"""
        self.fit_scaler(features)
        return self.transform(features)
    
    def get_top_kmers(
        self, 
        frequencies: np.ndarray, 
        n: int = 10
    ) -> List[Tuple[str, float]]:
        """Get top-n k-mers by frequency"""
        top_indices = np.argsort(frequencies)[-n:][::-1]
        return [(self.kmers[i], frequencies[i]) for i in top_indices]
    
    def compute_gc_content(self, sequence: str) -> float:
        """Compute GC content of sequence"""
        seq = sequence.upper()
        gc_count = seq.count('G') + seq.count('C')
        return gc_count / len(seq) if len(seq) > 0 else 0.0
    
    def extract_extended_features(self, sequence: str) -> np.ndarray:
        """
        Extract extended features including k-mers and additional metrics
        
        Args:
            sequence: DNA sequence
            
        Returns:
            Extended feature vector
        """
        # K-mer frequencies
        kmer_freq = self.extract(sequence)
        
        # Additional features
        gc_content = self.compute_gc_content(sequence)
        seq_length = len(sequence) / 1000  # Normalized length
        
        # Dinucleotide frequencies (important for taxonomy)
        dinuc_ext = KmerExtractor(k=2)
        dinuc_freq = dinuc_ext.extract(sequence)
        
        # Combine all features
        extended = np.concatenate([
            kmer_freq,
            dinuc_freq,
            [gc_content, seq_length]
        ])
        
        return extended
    
    def save(self, path: str):
        """Save extractor state"""
        state = {
            'k': self.k,
            'alphabet': self.alphabet,
            'normalize': self.normalize,
            'kmers': self.kmers,
            'fitted': self.fitted
        }
        if self.fitted:
            state['scaler_mean'] = self.scaler.mean_.tolist()
            state['scaler_scale'] = self.scaler.scale_.tolist()
        
        with open(path, 'w') as f:
            json.dump(state, f)
    
    @classmethod
    def load(cls, path: str) -> 'KmerExtractor':
        """Load extractor state"""
        with open(path, 'r') as f:
            state = json.load(f)
        
        extractor = cls(
            k=state['k'],
            alphabet=state['alphabet'],
            normalize=state['normalize']
        )
        
        if state['fitted']:
            extractor.scaler.mean_ = np.array(state['scaler_mean'])
            extractor.scaler.scale_ = np.array(state['scaler_scale'])
            extractor.fitted = True
        
        return extractor


class HybridCNNKmerModel:
    """
    Hybrid model combining CNN for sequence features 
    with MLP for k-mer frequencies
    """
    
    def __init__(
        self,
        input_length: int,
        num_classes: int,
        encoding_dim: int = 5,
        kmer_dim: int = 256,  # 4^4 = 256 for 4-mers
        architecture: str = 'attention_resnet'
    ):
        self.input_length = input_length
        self.num_classes = num_classes
        self.encoding_dim = encoding_dim
        self.kmer_dim = kmer_dim
        self.architecture = architecture
        
        self.model = self._build_model()
        self.history = None
        
    def _build_model(self) -> Model:
        """Build hybrid CNN + k-mer model"""
        
        # Input 1: Encoded sequences
        seq_input = keras.Input(shape=(self.input_length, self.encoding_dim), name='sequence_input')
        
        # Input 2: K-mer features
        kmer_input = keras.Input(shape=(self.kmer_dim,), name='kmer_input')
        
        # CNN branch for sequences
        x = layers.Conv1D(64, 7, padding='same')(seq_input)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling1D(2)(x)
        
        # Residual blocks
        for filters in [64, 128, 256]:
            x = ResidualBlock(filters)(x)
            x = layers.MaxPooling1D(2)(x)
            x = layers.Dropout(0.2)(x)
        
        # Attention
        x = ChannelAttention()(x)
        
        # Global pooling
        cnn_features = layers.GlobalAveragePooling1D()(x)
        
        # K-mer branch
        k = layers.Dense(128, activation='relu')(kmer_input)
        k = layers.BatchNormalization()(k)
        k = layers.Dropout(0.3)(k)
        k = layers.Dense(64, activation='relu')(k)
        kmer_features = layers.Dropout(0.3)(k)
        
        # Combine branches
        combined = layers.concatenate([cnn_features, kmer_features])
        
        # Classification head
        x = layers.Dense(256, activation='relu')(combined)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=[seq_input, kmer_input], outputs=outputs, name='hybrid_cnn_kmer')
        return model
    
    def compile(self, learning_rate: float = 0.001):
        """Compile the model"""
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy')
            ]
        )
    
    def train(
        self,
        X_seq_train: np.ndarray,
        X_kmer_train: np.ndarray,
        y_train: np.ndarray,
        X_seq_val: np.ndarray,
        X_kmer_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: int = 1
    ):
        """
        Train the hybrid model
        
        Args:
            X_seq_train: Encoded sequences (training)
            X_kmer_train: K-mer features (training)
            y_train: Labels (training)
            X_seq_val, X_kmer_val, y_val: Validation data
            epochs: Training epochs
            batch_size: Batch size
            verbose: Verbosity
        """
        if self.model.optimizer is None:
            self.compile()
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5
            )
        ]
        
        self.history = self.model.fit(
            {'sequence_input': X_seq_train, 'kmer_input': X_kmer_train},
            y_train,
            validation_data=(
                {'sequence_input': X_seq_val, 'kmer_input': X_kmer_val},
                y_val
            ),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return self.history
    
    def predict(
        self,
        X_seq: np.ndarray,
        X_kmer: np.ndarray,
        batch_size: int = 32
    ) -> np.ndarray:
        """Get predictions"""
        return self.model.predict(
            {'sequence_input': X_seq, 'kmer_input': X_kmer},
            batch_size=batch_size
        )
    
    def evaluate(
        self,
        X_seq: np.ndarray,
        X_kmer: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32
    ) -> Dict:
        """Evaluate model"""
        metrics = self.model.evaluate(
            {'sequence_input': X_seq, 'kmer_input': X_kmer},
            y,
            batch_size=batch_size,
            verbose=0
        )
        
        predictions = self.predict(X_seq, X_kmer, batch_size)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y, axis=1) if len(y.shape) > 1 else y
        
        return {
            'loss': metrics[0],
            'accuracy': metrics[1],
            'top3_accuracy': metrics[2],
            'predictions': predictions,
            'y_pred': y_pred,
            'y_true': y_true
        }
    
    def summary(self):
        """Print model summary"""
        self.model.summary()
    
    def save(self, path: str):
        """Save model"""
        self.model.save(path)
        
        config = {
            'input_length': self.input_length,
            'num_classes': self.num_classes,
            'encoding_dim': self.encoding_dim,
            'kmer_dim': self.kmer_dim,
            'architecture': self.architecture
        }
        
        config_path = Path(path).parent / 'hybrid_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'HybridCNNKmerModel':
        """Load model"""
        config_path = Path(path).parent / 'hybrid_config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        instance = cls(
            input_length=config['input_length'],
            num_classes=config['num_classes'],
            encoding_dim=config['encoding_dim'],
            kmer_dim=config['kmer_dim'],
            architecture=config['architecture']
        )
        
        instance.model = keras.models.load_model(
            path,
            custom_objects={
                'ResidualBlock': ResidualBlock,
                'ChannelAttention': ChannelAttention
            }
        )
        return instance


class MultiKmerExtractor:
    """
    Extract multiple k-mer sizes and combine
    """
    
    def __init__(self, k_values: List[int] = [3, 4, 5, 6]):
        self.k_values = k_values
        self.extractors = {k: KmerExtractor(k=k) for k in k_values}
        self.total_features = sum(4**k for k in k_values)
        
    def extract(self, sequence: str) -> np.ndarray:
        """Extract multi-scale k-mer features"""
        features = []
        for k in self.k_values:
            features.append(self.extractors[k].extract(sequence))
        return np.concatenate(features)
    
    def extract_batch(self, sequences: List[str]) -> np.ndarray:
        """Extract features for multiple sequences"""
        return np.array([self.extract(seq) for seq in sequences])


def create_kmer_features_from_encoded(
    X_encoded: np.ndarray,
    k: int = 4
) -> np.ndarray:
    """
    Create k-mer features from already encoded sequences
    
    Args:
        X_encoded: One-hot encoded sequences [n_samples, seq_len, 5]
        k: K-mer size
        
    Returns:
        K-mer frequency features
    """
    # Decode sequences first
    decoding = {0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: 'N'}
    
    sequences = []
    for x in X_encoded:
        indices = np.argmax(x, axis=1)
        seq = ''.join(decoding[i] for i in indices)
        sequences.append(seq)
    
    # Extract k-mer features
    extractor = KmerExtractor(k=k)
    return extractor.extract_batch(sequences)


def train_hybrid_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    sequences_train: List[str],
    sequences_val: List[str],
    input_length: int,
    num_classes: int,
    output_dir: str,
    k: int = 4,
    epochs: int = 50,
    batch_size: int = 32
) -> Dict:
    """
    Convenience function to train hybrid model
    
    Args:
        X_train, y_train: Encoded sequences and labels (training)
        X_val, y_val: Validation data
        sequences_train, sequences_val: Raw sequences for k-mer extraction
        input_length: Sequence length
        num_classes: Number of classes
        output_dir: Output directory
        k: K-mer size
        epochs: Training epochs
        batch_size: Batch size
        
    Returns:
        Training results
    """
    print("="*60)
    print("TRAINING HYBRID CNN + K-MER MODEL")
    print("="*60)
    
    # Extract k-mer features
    print(f"\nExtracting {k}-mer features...")
    extractor = KmerExtractor(k=k)
    
    X_kmer_train = extractor.extract_batch(sequences_train)
    X_kmer_val = extractor.extract_batch(sequences_val)
    
    # Standardize
    X_kmer_train = extractor.fit_transform(X_kmer_train)
    X_kmer_val = extractor.transform(X_kmer_val)
    
    print(f"K-mer feature dimension: {X_kmer_train.shape[1]}")
    
    # Create and train model
    model = HybridCNNKmerModel(
        input_length=input_length,
        num_classes=num_classes,
        kmer_dim=X_kmer_train.shape[1]
    )
    
    print("\nModel architecture:")
    model.summary()
    
    print("\nTraining...")
    history = model.train(
        X_train, X_kmer_train, y_train,
        X_val, X_kmer_val, y_val,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Evaluate
    metrics = model.evaluate(X_val, X_kmer_val, y_val)
    
    print(f"\nResults:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Top-3 Accuracy: {metrics['top3_accuracy']:.4f}")
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model.save(str(output_path / 'hybrid_model.h5'))
    extractor.save(str(output_path / 'kmer_extractor.json'))
    
    results = {
        'accuracy': float(metrics['accuracy']),
        'top3_accuracy': float(metrics['top3_accuracy']),
        'k': k,
        'kmer_dim': X_kmer_train.shape[1]
    }
    
    with open(output_path / 'hybrid_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == '__main__':
    print("K-mer Feature Extraction Module")
    print("="*60)
    
    # Test with sample sequence
    test_seq = "ATCGATCGATCGATCGATCG" * 25
    print(f"Test sequence length: {len(test_seq)}")
    
    # Test k-mer extraction
    for k in [3, 4, 5, 6]:
        extractor = KmerExtractor(k=k)
        features = extractor.extract(test_seq)
        print(f"\n{k}-mer features: {len(features)} dimensions")
        
        # Top k-mers
        top = extractor.get_top_kmers(features, n=5)
        print(f"Top 5 {k}-mers:")
        for kmer, freq in top:
            print(f"  {kmer}: {freq:.4f}")
    
    print("\n" + "="*60)
    print("Usage:")
    print("  from kmer_features import KmerExtractor, HybridCNNKmerModel")
    print("  extractor = KmerExtractor(k=4)")
    print("  kmer_features = extractor.extract_batch(sequences)")
    print("  model = HybridCNNKmerModel(input_length=500, num_classes=126)")
    print("  model.train(X_seq, X_kmer, y, ...)")
