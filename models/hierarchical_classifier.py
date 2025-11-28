"""
Hierarchical Classification System for eDNA Taxonomy
Train separate models for each taxonomic level
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
from collections import defaultdict
import pickle

from improved_cnn_classifier import ImprovedCNNClassifier, create_improved_model


class TaxonomyTree:
    """
    Hierarchical taxonomy structure for organizing classification
    """
    
    def __init__(self):
        self.tree = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
        self.level_to_idx = {}
        self.idx_to_level = {}
        
    def add_taxon(
        self, 
        domain: str, 
        phylum: str, 
        class_name: str, 
        order: str = None,
        family: str = None
    ):
        """Add a taxonomic path to the tree"""
        self.tree[domain][phylum][class_name].add(order or 'Unknown')
        
    def build_from_labels(self, taxonomy_labels: List[str], separator: str = ';'):
        """
        Build tree from full taxonomy strings
        
        Args:
            taxonomy_labels: List of taxonomy strings like "Bacteria;Proteobacteria;Gammaproteobacteria"
            separator: Separator between levels
        """
        for label in taxonomy_labels:
            parts = label.split(separator)
            if len(parts) >= 3:
                domain = parts[0].strip()
                phylum = parts[1].strip()
                class_name = parts[2].strip()
                order = parts[3].strip() if len(parts) > 3 else None
                self.add_taxon(domain, phylum, class_name, order)
        
        self._build_indices()
    
    def _build_indices(self):
        """Build mapping between names and indices for each level"""
        # Domain level
        domains = sorted(self.tree.keys())
        self.level_to_idx['domain'] = {d: i for i, d in enumerate(domains)}
        self.idx_to_level['domain'] = {i: d for i, d in enumerate(domains)}
        
        # Phylum level (per domain)
        self.level_to_idx['phylum'] = {}
        self.idx_to_level['phylum'] = {}
        
        for domain in domains:
            phyla = sorted(self.tree[domain].keys())
            self.level_to_idx['phylum'][domain] = {p: i for i, p in enumerate(phyla)}
            self.idx_to_level['phylum'][domain] = {i: p for i, p in enumerate(phyla)}
        
        # Class level (per phylum)
        self.level_to_idx['class'] = {}
        self.idx_to_level['class'] = {}
        
        for domain in domains:
            self.level_to_idx['class'][domain] = {}
            self.idx_to_level['class'][domain] = {}
            for phylum in self.tree[domain]:
                classes = sorted(self.tree[domain][phylum].keys())
                self.level_to_idx['class'][domain][phylum] = {c: i for i, c in enumerate(classes)}
                self.idx_to_level['class'][domain][phylum] = {i: c for i, c in enumerate(classes)}
    
    def get_num_classes(self, level: str, parent: str = None, grandparent: str = None) -> int:
        """Get number of classes at a given level"""
        if level == 'domain':
            return len(self.tree)
        elif level == 'phylum':
            return len(self.tree[parent])
        elif level == 'class':
            return len(self.tree[grandparent][parent])
        return 0
    
    def encode_label(
        self, 
        taxonomy: str, 
        level: str,
        separator: str = ';'
    ) -> int:
        """Encode taxonomy string to class index at given level"""
        parts = taxonomy.split(separator)
        parts = [p.strip() for p in parts]
        
        if level == 'domain':
            return self.level_to_idx['domain'].get(parts[0], -1)
        elif level == 'phylum':
            if parts[0] in self.level_to_idx['phylum']:
                return self.level_to_idx['phylum'][parts[0]].get(parts[1], -1)
        elif level == 'class':
            if parts[0] in self.level_to_idx['class']:
                if parts[1] in self.level_to_idx['class'][parts[0]]:
                    return self.level_to_idx['class'][parts[0]][parts[1]].get(parts[2], -1)
        return -1
    
    def decode_label(
        self, 
        idx: int, 
        level: str,
        parent: str = None,
        grandparent: str = None
    ) -> str:
        """Decode class index to taxonomy name"""
        if level == 'domain':
            return self.idx_to_level['domain'].get(idx, 'Unknown')
        elif level == 'phylum':
            if parent in self.idx_to_level['phylum']:
                return self.idx_to_level['phylum'][parent].get(idx, 'Unknown')
        elif level == 'class':
            if grandparent in self.idx_to_level['class']:
                if parent in self.idx_to_level['class'][grandparent]:
                    return self.idx_to_level['class'][grandparent][parent].get(idx, 'Unknown')
        return 'Unknown'
    
    def save(self, path: str):
        """Save taxonomy tree to file"""
        data = {
            'tree': dict(self.tree),
            'level_to_idx': self.level_to_idx,
            'idx_to_level': self.idx_to_level
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, path: str) -> 'TaxonomyTree':
        """Load taxonomy tree from file"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        tree = cls()
        tree.tree = defaultdict(lambda: defaultdict(lambda: defaultdict(set)), data['tree'])
        tree.level_to_idx = data['level_to_idx']
        tree.idx_to_level = data['idx_to_level']
        return tree


class HierarchicalClassifier:
    """
    Hierarchical classification with separate models for each taxonomic level
    """
    
    def __init__(
        self,
        input_length: int,
        encoding_dim: int = 5,
        architecture: str = 'attention_resnet'
    ):
        self.input_length = input_length
        self.encoding_dim = encoding_dim
        self.architecture = architecture
        
        self.taxonomy = TaxonomyTree()
        self.models = {}
        self.trained_levels = set()
    
    def prepare_data(
        self,
        X: np.ndarray,
        taxonomy_labels: List[str],
        level: str,
        parent_filter: Optional[Dict[str, str]] = None
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Prepare data for training at a specific level
        
        Args:
            X: Encoded sequences
            taxonomy_labels: Full taxonomy strings
            level: Taxonomic level to prepare for
            parent_filter: Filter by parent taxonomy
            
        Returns:
            Filtered X, one-hot encoded y, number of classes
        """
        if not self.taxonomy.level_to_idx:
            self.taxonomy.build_from_labels(taxonomy_labels)
        
        # Filter and encode labels
        valid_indices = []
        encoded_labels = []
        
        for i, tax in enumerate(taxonomy_labels):
            if parent_filter:
                # Check if matches parent filter
                parts = tax.split(';')
                if level == 'phylum' and parts[0].strip() != parent_filter.get('domain'):
                    continue
                if level == 'class':
                    if parts[0].strip() != parent_filter.get('domain'):
                        continue
                    if parts[1].strip() != parent_filter.get('phylum'):
                        continue
            
            label_idx = self.taxonomy.encode_label(tax, level)
            if label_idx >= 0:
                valid_indices.append(i)
                encoded_labels.append(label_idx)
        
        X_filtered = X[valid_indices]
        y_encoded = np.array(encoded_labels)
        
        # Get number of classes
        if level == 'domain':
            num_classes = self.taxonomy.get_num_classes('domain')
        elif level == 'phylum':
            num_classes = self.taxonomy.get_num_classes('phylum', parent_filter['domain'])
        elif level == 'class':
            num_classes = self.taxonomy.get_num_classes('class', 
                                                        parent_filter['phylum'],
                                                        parent_filter['domain'])
        
        # One-hot encode
        y_onehot = keras.utils.to_categorical(y_encoded, num_classes)
        
        return X_filtered, y_onehot, num_classes
    
    def train_level(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        taxonomy_train: List[str],
        taxonomy_val: List[str],
        level: str,
        parent_filter: Optional[Dict[str, str]] = None,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: int = 1
    ) -> Dict:
        """
        Train model for a specific taxonomic level
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data  
            taxonomy_train, taxonomy_val: Taxonomy labels
            level: 'domain', 'phylum', or 'class'
            parent_filter: Filter by parent taxonomy
            epochs: Training epochs
            batch_size: Batch size
            verbose: Verbosity level
            
        Returns:
            Training results
        """
        print(f"\n{'='*60}")
        print(f"Training {level.upper()} level classifier")
        if parent_filter:
            print(f"  Parent filter: {parent_filter}")
        print('='*60)
        
        # Prepare data for this level
        X_train_level, y_train_level, num_classes = self.prepare_data(
            X_train, taxonomy_train, level, parent_filter
        )
        X_val_level, y_val_level, _ = self.prepare_data(
            X_val, taxonomy_val, level, parent_filter
        )
        
        print(f"Training samples: {len(X_train_level)}")
        print(f"Validation samples: {len(X_val_level)}")
        print(f"Number of classes: {num_classes}")
        
        if num_classes < 2:
            print(f"Skipping - not enough classes")
            return None
        
        # Create model
        model = create_improved_model(
            input_length=self.input_length,
            num_classes=num_classes,
            encoding_dim=self.encoding_dim,
            architecture=self.architecture
        )
        
        # Train
        history = model.train(
            X_train_level, y_train_level,
            X_val_level, y_val_level,
            epochs=epochs,
            batch_size=batch_size,
            use_class_weights=True,
            verbose=verbose
        )
        
        # Evaluate
        metrics = model.evaluate(X_val_level, y_val_level)
        
        # Store model
        model_key = self._get_model_key(level, parent_filter)
        self.models[model_key] = model
        self.trained_levels.add(level)
        
        results = {
            'level': level,
            'parent_filter': parent_filter,
            'num_classes': num_classes,
            'accuracy': float(metrics['accuracy']),
            'top3_accuracy': float(metrics.get('top3_accuracy', 0)),
            'train_samples': len(X_train_level),
            'val_samples': len(X_val_level)
        }
        
        print(f"\nResults for {level}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Top-3 Accuracy: {metrics.get('top3_accuracy', 0):.4f}")
        
        return results
    
    def train_all_levels(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        taxonomy_train: List[str],
        taxonomy_val: List[str],
        min_samples_per_class: int = 10,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: int = 1
    ) -> Dict:
        """
        Train all hierarchical levels
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            taxonomy_train, taxonomy_val: Taxonomy labels
            min_samples_per_class: Minimum samples required
            epochs: Training epochs
            batch_size: Batch size
            verbose: Verbosity level
            
        Returns:
            Results for all levels
        """
        # Build taxonomy tree
        self.taxonomy.build_from_labels(taxonomy_train)
        
        all_results = {}
        
        # Level 1: Domain
        result = self.train_level(
            X_train, y_train, X_val, y_val,
            taxonomy_train, taxonomy_val,
            level='domain',
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )
        if result:
            all_results['domain'] = result
        
        # Level 2: Phylum (for each domain)
        all_results['phylum'] = {}
        for domain in self.taxonomy.tree.keys():
            result = self.train_level(
                X_train, y_train, X_val, y_val,
                taxonomy_train, taxonomy_val,
                level='phylum',
                parent_filter={'domain': domain},
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbose
            )
            if result:
                all_results['phylum'][domain] = result
        
        # Level 3: Class (for each domain-phylum combination)
        all_results['class'] = {}
        for domain in self.taxonomy.tree.keys():
            all_results['class'][domain] = {}
            for phylum in self.taxonomy.tree[domain].keys():
                result = self.train_level(
                    X_train, y_train, X_val, y_val,
                    taxonomy_train, taxonomy_val,
                    level='class',
                    parent_filter={'domain': domain, 'phylum': phylum},
                    epochs=epochs // 2,  # Fewer epochs for lower levels
                    batch_size=batch_size,
                    verbose=verbose
                )
                if result:
                    all_results['class'][domain][phylum] = result
        
        return all_results
    
    def predict(
        self,
        X: np.ndarray,
        return_all_levels: bool = True,
        confidence_threshold: float = 0.5
    ) -> Dict:
        """
        Hierarchical prediction
        
        Args:
            X: Encoded sequences to classify
            return_all_levels: Return predictions at all levels
            confidence_threshold: Minimum confidence to proceed to next level
            
        Returns:
            Predictions at each level
        """
        predictions = {
            'domain': [],
            'domain_prob': [],
            'phylum': [],
            'phylum_prob': [],
            'class': [],
            'class_prob': [],
            'full_taxonomy': []
        }
        
        # Level 1: Domain
        domain_model = self.models.get('domain')
        if domain_model is None:
            raise ValueError("Domain model not trained")
        
        domain_probs = domain_model.predict(X)
        domain_preds = np.argmax(domain_probs, axis=1)
        domain_confs = np.max(domain_probs, axis=1)
        
        for i in range(len(X)):
            domain_name = self.taxonomy.decode_label(domain_preds[i], 'domain')
            predictions['domain'].append(domain_name)
            predictions['domain_prob'].append(float(domain_confs[i]))
            
            # Level 2: Phylum
            if domain_confs[i] >= confidence_threshold:
                phylum_key = f"phylum_{domain_name}"
                phylum_model = self.models.get(phylum_key)
                
                if phylum_model:
                    phylum_prob = phylum_model.predict(X[i:i+1])[0]
                    phylum_pred = np.argmax(phylum_prob)
                    phylum_conf = np.max(phylum_prob)
                    phylum_name = self.taxonomy.decode_label(
                        phylum_pred, 'phylum', parent=domain_name
                    )
                else:
                    phylum_name = 'Unknown'
                    phylum_conf = 0.0
            else:
                phylum_name = 'Unknown'
                phylum_conf = 0.0
            
            predictions['phylum'].append(phylum_name)
            predictions['phylum_prob'].append(float(phylum_conf))
            
            # Level 3: Class
            if phylum_conf >= confidence_threshold:
                class_key = f"class_{domain_name}_{phylum_name}"
                class_model = self.models.get(class_key)
                
                if class_model:
                    class_prob = class_model.predict(X[i:i+1])[0]
                    class_pred = np.argmax(class_prob)
                    class_conf = np.max(class_prob)
                    class_name = self.taxonomy.decode_label(
                        class_pred, 'class', 
                        parent=phylum_name, 
                        grandparent=domain_name
                    )
                else:
                    class_name = 'Unknown'
                    class_conf = 0.0
            else:
                class_name = 'Unknown'
                class_conf = 0.0
            
            predictions['class'].append(class_name)
            predictions['class_prob'].append(float(class_conf))
            
            # Full taxonomy
            full_tax = f"{domain_name};{phylum_name};{class_name}"
            predictions['full_taxonomy'].append(full_tax)
        
        return predictions
    
    def _get_model_key(self, level: str, parent_filter: Optional[Dict] = None) -> str:
        """Generate unique key for model storage"""
        if level == 'domain':
            return 'domain'
        elif level == 'phylum':
            return f"phylum_{parent_filter['domain']}"
        elif level == 'class':
            return f"class_{parent_filter['domain']}_{parent_filter['phylum']}"
        return level
    
    def save(self, output_dir: str):
        """Save all models and taxonomy"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save taxonomy
        self.taxonomy.save(str(output_path / 'taxonomy.pkl'))
        
        # Save models
        models_dir = output_path / 'models'
        models_dir.mkdir(exist_ok=True)
        
        model_paths = {}
        for key, model in self.models.items():
            model_path = str(models_dir / f'{key}.h5')
            model.save(model_path)
            model_paths[key] = model_path
        
        # Save config
        config = {
            'input_length': self.input_length,
            'encoding_dim': self.encoding_dim,
            'architecture': self.architecture,
            'trained_levels': list(self.trained_levels),
            'model_paths': model_paths
        }
        with open(output_path / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Hierarchical classifier saved to {output_dir}")
    
    @classmethod
    def load(cls, input_dir: str) -> 'HierarchicalClassifier':
        """Load hierarchical classifier"""
        input_path = Path(input_dir)
        
        # Load config
        with open(input_path / 'config.json', 'r') as f:
            config = json.load(f)
        
        classifier = cls(
            input_length=config['input_length'],
            encoding_dim=config['encoding_dim'],
            architecture=config['architecture']
        )
        
        # Load taxonomy
        classifier.taxonomy = TaxonomyTree.load(str(input_path / 'taxonomy.pkl'))
        
        # Load models
        for key, path in config['model_paths'].items():
            classifier.models[key] = ImprovedCNNClassifier.load(path)
        
        classifier.trained_levels = set(config['trained_levels'])
        
        return classifier


def train_hierarchical_classifier(
    X_train: np.ndarray,
    X_val: np.ndarray,
    taxonomy_train: List[str],
    taxonomy_val: List[str],
    input_length: int,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 32
) -> HierarchicalClassifier:
    """
    Convenience function to train complete hierarchical classifier
    """
    classifier = HierarchicalClassifier(
        input_length=input_length,
        architecture='attention_resnet'
    )
    
    results = classifier.train_all_levels(
        X_train, None, X_val, None,
        taxonomy_train, taxonomy_val,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    classifier.save(output_dir)
    
    return classifier


if __name__ == '__main__':
    print("Hierarchical Classification Module")
    print("="*60)
    print("This module provides multi-level taxonomic classification:")
    print("  Level 1: Domain (Bacteria, Archaea, Eukaryota)")
    print("  Level 2: Phylum (within each domain)")
    print("  Level 3: Class (within each phylum)")
    print()
    print("Usage:")
    print("  from hierarchical_classifier import HierarchicalClassifier")
    print("  classifier = HierarchicalClassifier(input_length=500)")
    print("  classifier.train_all_levels(X_train, y_train, X_val, y_val, tax_train, tax_val)")
    print("  predictions = classifier.predict(X_test)")
