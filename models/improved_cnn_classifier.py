"""
Improved CNN Classifier for Deep-Sea eDNA Classification
Features: Residual blocks, Attention mechanism, Multi-scale convolutions
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
)
from sklearn.utils.class_weight import compute_class_weight
from typing import Optional, Dict, List, Tuple
import json
from pathlib import Path


class ResidualBlock(layers.Layer):
    """Residual block with skip connections for better gradient flow"""
    
    def __init__(self, filters: int, kernel_size: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        
    def build(self, input_shape):
        self.conv1 = layers.Conv1D(self.filters, self.kernel_size, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv1D(self.filters, self.kernel_size, padding='same')
        self.bn2 = layers.BatchNormalization()
        
        # Shortcut projection if dimensions don't match
        if input_shape[-1] != self.filters:
            self.shortcut = layers.Conv1D(self.filters, 1, padding='same')
        else:
            self.shortcut = lambda x: x
            
        self.add = layers.Add()
        self.activation = layers.Activation('relu')
        
    def call(self, inputs, training=False):
        shortcut = self.shortcut(inputs)
        
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        x = self.add([shortcut, x])
        x = self.activation(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size
        })
        return config


class SelfAttention(layers.Layer):
    """Self-attention layer for capturing long-range dependencies in sequences"""
    
    def __init__(self, units: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        
    def build(self, input_shape):
        self.query = layers.Dense(self.units)
        self.key = layers.Dense(self.units)
        self.value = layers.Dense(self.units)
        self.output_dense = layers.Dense(input_shape[-1])
        
    def call(self, inputs):
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)
        
        # Scaled dot-product attention
        scores = tf.matmul(q, k, transpose_b=True)
        scores = scores / tf.math.sqrt(tf.cast(self.units, tf.float32))
        attention_weights = tf.nn.softmax(scores, axis=-1)
        
        attended = tf.matmul(attention_weights, v)
        output = self.output_dense(attended)
        
        return output + inputs  # Residual connection
    
    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config


class ChannelAttention(layers.Layer):
    """Channel-wise attention (Squeeze-and-Excitation style)"""
    
    def __init__(self, reduction_ratio: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        
    def build(self, input_shape):
        channels = input_shape[-1]
        self.fc1 = layers.Dense(channels // self.reduction_ratio, activation='relu')
        self.fc2 = layers.Dense(channels, activation='sigmoid')
        
    def call(self, inputs):
        gap = tf.reduce_mean(inputs, axis=1, keepdims=True)
        attention = self.fc1(gap)
        attention = self.fc2(attention)
        return inputs * attention
    
    def get_config(self):
        config = super().get_config()
        config.update({'reduction_ratio': self.reduction_ratio})
        return config


class MultiScaleConv(layers.Layer):
    """Multi-scale convolution for capturing patterns at different scales"""
    
    def __init__(self, filters: int, kernel_sizes: List[int] = [3, 5, 7, 11], **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        
    def build(self, input_shape):
        self.convs = [
            layers.Conv1D(self.filters // len(self.kernel_sizes), k, padding='same')
            for k in self.kernel_sizes
        ]
        self.bn = layers.BatchNormalization()
        
    def call(self, inputs, training=False):
        outputs = [conv(inputs) for conv in self.convs]
        x = layers.concatenate(outputs, axis=-1)
        x = self.bn(x, training=training)
        return tf.nn.relu(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_sizes': self.kernel_sizes
        })
        return config


class ImprovedCNNClassifier:
    """
    Enhanced CNN classifier with:
    - Residual connections
    - Self-attention mechanism
    - Multi-scale convolutions
    - Class weighting for imbalanced data
    """
    
    def __init__(
        self,
        input_length: int,
        num_classes: int,
        encoding_dim: int = 5,
        architecture: str = 'attention_resnet'
    ):
        self.input_length = input_length
        self.num_classes = num_classes
        self.encoding_dim = encoding_dim
        self.architecture = architecture
        self.model = self._build_model()
        self.class_weights = None
        self.history = None
        
    def _build_model(self) -> Model:
        """Build the model based on selected architecture"""
        if self.architecture == 'attention_resnet':
            return self._build_attention_resnet()
        elif self.architecture == 'multiscale':
            return self._build_multiscale_model()
        elif self.architecture == 'deep_resnet':
            return self._build_deep_resnet()
        else:
            return self._build_attention_resnet()
            
    def _build_attention_resnet(self) -> Model:
        """ResNet with attention - balanced performance and complexity"""
        inputs = keras.Input(shape=(self.input_length, self.encoding_dim))
        
        # Initial convolution
        x = layers.Conv1D(64, 7, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling1D(2)(x)
        
        # Residual blocks with increasing filters
        for filters in [64, 128, 256]:
            x = ResidualBlock(filters)(x)
            x = layers.MaxPooling1D(2)(x)
            x = layers.Dropout(0.2)(x)
        
        # Channel attention
        x = ChannelAttention()(x)
        
        # Self-attention for long-range dependencies
        x = SelfAttention(units=64)(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Classification head
        x = layers.Dense(256, activation='relu', 
                        kernel_regularizer=regularizers.l2(0.01))(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs, name='attention_resnet')
        return model
    
    def _build_multiscale_model(self) -> Model:
        """Multi-scale CNN for capturing patterns at different resolutions"""
        inputs = keras.Input(shape=(self.input_length, self.encoding_dim))
        
        # Multi-scale initial features
        x = MultiScaleConv(64, kernel_sizes=[3, 5, 7, 11])(inputs)
        x = layers.MaxPooling1D(2)(x)
        
        # Residual blocks
        for filters in [128, 256]:
            x = ResidualBlock(filters)(x)
            x = layers.MaxPooling1D(2)(x)
            x = layers.Dropout(0.25)(x)
        
        # Attention
        x = ChannelAttention()(x)
        
        # Global features
        avg_pool = layers.GlobalAveragePooling1D()(x)
        max_pool = layers.GlobalMaxPooling1D()(x)
        x = layers.concatenate([avg_pool, max_pool])
        
        # Classification
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs, name='multiscale_cnn')
        return model
    
    def _build_deep_resnet(self) -> Model:
        """Deeper ResNet for complex classification tasks"""
        inputs = keras.Input(shape=(self.input_length, self.encoding_dim))
        
        # Stem
        x = layers.Conv1D(64, 7, strides=2, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling1D(3, strides=2, padding='same')(x)
        
        # Residual stages
        filter_configs = [
            (64, 2),
            (128, 3),
            (256, 4),
            (512, 2),
        ]
        
        for filters, num_blocks in filter_configs:
            for i in range(num_blocks):
                x = ResidualBlock(filters)(x)
            x = layers.MaxPooling1D(2, padding='same')(x)
            x = layers.Dropout(0.2)(x)
        
        # Attention layers
        x = ChannelAttention()(x)
        x = SelfAttention(units=128)(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Classification
        x = layers.Dense(512, activation='relu',
                        kernel_regularizer=regularizers.l2(0.01))(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs, name='deep_resnet')
        return model
    
    def compute_class_weights(self, y_train: np.ndarray) -> Dict[int, float]:
        """Compute balanced class weights for imbalanced datasets"""
        if len(y_train.shape) > 1:
            y_indices = np.argmax(y_train, axis=1)
        else:
            y_indices = y_train
            
        classes = np.unique(y_indices)
        weights = compute_class_weight('balanced', classes=classes, y=y_indices)
        self.class_weights = dict(enumerate(weights))
        
        print(f"Computed class weights for {len(classes)} classes")
        print(f"  Min weight: {min(weights):.4f}")
        print(f"  Max weight: {max(weights):.4f}")
        print(f"  Mean weight: {np.mean(weights):.4f}")
        
        return self.class_weights
    
    def compile(
        self,
        learning_rate: float = 0.001,
        label_smoothing: float = 0.1
    ):
        """Compile model with optimized settings"""
        optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        loss = keras.losses.CategoricalCrossentropy(
            label_smoothing=label_smoothing
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[
                'accuracy',
                keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy'),
                keras.metrics.TopKCategoricalAccuracy(k=5, name='top5_accuracy'),
            ]
        )
        
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        use_class_weights: bool = True,
        checkpoint_path: Optional[str] = None,
        verbose: int = 1
    ) -> keras.callbacks.History:
        """Train the model with advanced callbacks"""
        
        if self.model.optimizer is None:
            self.compile()
        
        if use_class_weights:
            self.compute_class_weights(y_train)
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
        ]
        
        if checkpoint_path:
            callbacks.append(
                ModelCheckpoint(
                    checkpoint_path,
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                )
            )
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=self.class_weights if use_class_weights else None,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return self.history
    
    def predict(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """Get probability predictions"""
        return self.model.predict(X, batch_size=batch_size)
    
    def predict_classes(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """Get class predictions"""
        probs = self.predict(X, batch_size)
        return np.argmax(probs, axis=1)
    
    def predict_top_k(
        self, 
        X: np.ndarray, 
        k: int = 3, 
        batch_size: int = 32
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get top-k predictions with probabilities"""
        probs = self.predict(X, batch_size)
        top_k_indices = np.argsort(probs, axis=1)[:, -k:][:, ::-1]
        top_k_probs = np.take_along_axis(probs, top_k_indices, axis=1)
        return top_k_indices, top_k_probs
    
    def evaluate(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        batch_size: int = 32
    ) -> Dict:
        """Evaluate model and return detailed metrics"""
        metrics = self.model.evaluate(X, y, batch_size=batch_size, verbose=0)
        
        predictions = self.predict(X, batch_size)
        y_pred = np.argmax(predictions, axis=1)
        y_true = np.argmax(y, axis=1) if len(y.shape) > 1 else y
        
        return {
            'loss': metrics[0],
            'accuracy': metrics[1],
            'top3_accuracy': metrics[2],
            'top5_accuracy': metrics[3],
            'predictions': predictions,
            'y_pred': y_pred,
            'y_true': y_true
        }
    
    def summary(self):
        """Print model summary"""
        self.model.summary()
        
    def save(self, path: str):
        """Save model and config"""
        self.model.save(path)
        
        config = {
            'input_length': self.input_length,
            'num_classes': self.num_classes,
            'encoding_dim': self.encoding_dim,
            'architecture': self.architecture
        }
        config_path = Path(path).parent / 'model_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
    @classmethod
    def load(cls, path: str) -> 'ImprovedCNNClassifier':
        """Load model from file"""
        config_path = Path(path).parent / 'model_config.json'
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        instance = cls(
            input_length=config['input_length'],
            num_classes=config['num_classes'],
            encoding_dim=config['encoding_dim'],
            architecture=config['architecture']
        )
        instance.model = keras.models.load_model(
            path,
            custom_objects={
                'ResidualBlock': ResidualBlock,
                'SelfAttention': SelfAttention,
                'ChannelAttention': ChannelAttention,
                'MultiScaleConv': MultiScaleConv
            }
        )
        return instance


def create_improved_model(
    input_length: int,
    num_classes: int,
    encoding_dim: int = 5,
    architecture: str = 'attention_resnet'
) -> ImprovedCNNClassifier:
    """Factory function to create improved CNN classifier"""
    model = ImprovedCNNClassifier(
        input_length=input_length,
        num_classes=num_classes,
        encoding_dim=encoding_dim,
        architecture=architecture
    )
    model.compile()
    return model
