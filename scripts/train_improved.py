#!/usr/bin/env python3
"""
Improved Training Script for Deep-Sea eDNA Classifier
Integrates all improvements: class weights, augmentation, improved architectures,
hierarchical classification, ensemble methods, and k-mer features
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# TensorFlow configuration
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.utils import to_categorical

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import modules
from models.improved_cnn_classifier import ImprovedCNNClassifier, create_improved_model
from models.ensemble_classifier import EnsembleClassifier, train_ensemble
from models.hierarchical_classifier import HierarchicalClassifier
from models.kmer_features import KmerExtractor, HybridCNNKmerModel
from utils.augmentation import AugmentationPipeline, DNAAugmenter
from utils.evaluation import ModelEvaluator, AbundanceEstimator, print_summary


def set_seeds(seed: int = 42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_dataset(data_dir: str):
    """Load preprocessed dataset"""
    data_path = Path(data_dir)
    
    print("Loading dataset...")
    train_data = np.load(data_path / 'train_data.npy')
    val_data = np.load(data_path / 'val_data.npy')
    test_data = np.load(data_path / 'test_data.npy')
    train_labels = np.load(data_path / 'train_labels.npy')
    val_labels = np.load(data_path / 'val_labels.npy')
    test_labels = np.load(data_path / 'test_labels.npy')
    
    with open(data_path / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    print(f"  Training samples: {len(train_data)}")
    print(f"  Validation samples: {len(val_data)}")
    print(f"  Test samples: {len(test_data)}")
    print(f"  Number of classes: {metadata['num_classes']}")
    
    return (train_data, val_data, test_data,
            train_labels, val_labels, test_labels, metadata)


def train_improved_classifier(args):
    """Train improved CNN classifier with all enhancements"""
    print("\n" + "="*70)
    print("TRAINING IMPROVED CNN CLASSIFIER")
    print("="*70)
    
    set_seeds(args.seed)
    
    # Load data
    train_data, val_data, test_data, train_labels, val_labels, test_labels, metadata = \
        load_dataset(args.data_dir)
    
    num_classes = metadata['num_classes']
    input_length = train_data.shape[1]
    encoding_dim = train_data.shape[2] if len(train_data.shape) > 2 else 5
    
    # Convert labels to categorical
    train_labels_cat = to_categorical(train_labels, num_classes)
    val_labels_cat = to_categorical(val_labels, num_classes)
    test_labels_cat = to_categorical(test_labels, num_classes)
    
    # Data augmentation
    if args.augment:
        print("\nApplying data augmentation...")
        pipeline = AugmentationPipeline(seed=args.seed)
        train_data_aug, train_labels_aug = pipeline.augment_dataset(
            train_data, train_labels_cat,
            augmentation_factor=args.augment_factor,
            use_mixup=args.use_mixup
        )
        print(f"  Augmented training size: {len(train_data_aug)}")
    else:
        train_data_aug = train_data
        train_labels_aug = train_labels_cat
    
    # Create model
    print(f"\nCreating {args.architecture} model...")
    model = create_improved_model(
        input_length=input_length,
        num_classes=num_classes,
        encoding_dim=encoding_dim,
        architecture=args.architecture
    )
    
    print("\nModel architecture:")
    model.summary()
    
    # Train
    print("\nTraining model...")
    print(f"  Class weights: {'Enabled' if args.use_class_weights else 'Disabled'}")
    print(f"  Label smoothing: {args.label_smoothing}")
    
    model.compile(
        learning_rate=args.learning_rate,
        label_smoothing=args.label_smoothing
    )
    
    checkpoint_path = str(Path(args.output_dir) / 'best_model.h5') if args.save_best else None
    
    history = model.train(
        train_data_aug, train_labels_aug,
        val_data, val_labels_cat,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_class_weights=args.use_class_weights,
        checkpoint_path=checkpoint_path,
        verbose=1
    )
    
    # Evaluate
    print("\nEvaluating on test set...")
    metrics = model.evaluate(test_data, test_labels_cat, batch_size=args.batch_size)
    
    # Get class names
    label_mapping = metadata.get('label_mapping', {})
    class_names = [label_mapping.get(str(i), f"Class_{i}") for i in range(num_classes)]
    
    # Detailed evaluation
    evaluator = ModelEvaluator(class_names=class_names)
    full_metrics = evaluator.compute_metrics(
        metrics['y_true'],
        metrics['y_pred'],
        metrics['predictions']
    )
    
    print_summary(full_metrics)
    
    # Generate report
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    evaluator.generate_report(
        metrics['y_true'],
        metrics['y_pred'],
        metrics['predictions'],
        output_dir=str(output_path / 'evaluation')
    )
    
    evaluator.plot_training_history(
        history,
        save_path=str(output_path / 'training_history.png')
    )
    
    # Abundance estimation
    abundance_estimator = AbundanceEstimator()
    abundance = abundance_estimator.estimate_from_probabilities(
        metrics['predictions'],
        class_names,
        threshold=0.5
    )
    
    print("\nTop 10 Predicted Taxonomic Abundance:")
    for i, (taxon, abund) in enumerate(list(abundance.items())[:10]):
        print(f"  {i+1}. {taxon:30s}: {abund*100:.2f}%")
    
    abundance_estimator.plot_abundance(
        top_n=20,
        save_path=str(output_path / 'abundance.png')
    )
    
    # Save model
    model.save(str(output_path / 'improved_classifier.h5'))
    
    # Save results
    results = {
        'model': 'improved_cnn_classifier',
        'architecture': args.architecture,
        'accuracy': float(full_metrics['accuracy']),
        'f1_macro': float(full_metrics['f1_macro']),
        'f1_weighted': float(full_metrics['f1_weighted']),
        'top3_accuracy': float(full_metrics.get('top3_accuracy', 0)),
        'top5_accuracy': float(full_metrics.get('top5_accuracy', 0)),
        'num_weak_classes': full_metrics['num_weak_classes'],
        'num_classes': num_classes,
        'augmentation': args.augment,
        'class_weights': args.use_class_weights
    }
    
    with open(output_path / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    return results


def train_ensemble_models(args):
    """Train ensemble of models"""
    print("\n" + "="*70)
    print("TRAINING ENSEMBLE CLASSIFIER")
    print("="*70)
    
    set_seeds(args.seed)
    
    # Load data
    train_data, val_data, test_data, train_labels, val_labels, test_labels, metadata = \
        load_dataset(args.data_dir)
    
    num_classes = metadata['num_classes']
    input_length = train_data.shape[1]
    
    train_labels_cat = to_categorical(train_labels, num_classes)
    val_labels_cat = to_categorical(val_labels, num_classes)
    test_labels_cat = to_categorical(test_labels, num_classes)
    
    # Train ensemble
    output_path = Path(args.output_dir) / 'ensemble'
    
    results = train_ensemble(
        train_data, train_labels_cat,
        val_data, val_labels_cat,
        test_data, test_labels_cat,
        input_length=input_length,
        num_classes=num_classes,
        output_dir=str(output_path),
        n_models=args.n_ensemble_models,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    print(f"\nEnsemble saved to {output_path}")
    
    return results


def train_hierarchical(args):
    """Train hierarchical classification system"""
    print("\n" + "="*70)
    print("TRAINING HIERARCHICAL CLASSIFIER")
    print("="*70)
    
    set_seeds(args.seed)
    
    # Load data
    train_data, val_data, test_data, train_labels, val_labels, test_labels, metadata = \
        load_dataset(args.data_dir)
    
    # Check for taxonomy labels
    taxonomy_file = Path(args.data_dir) / 'taxonomy_labels.json'
    if not taxonomy_file.exists():
        print("ERROR: Hierarchical training requires taxonomy labels.")
        print(f"Expected file: {taxonomy_file}")
        print("\nPlease run prepare_dataset.py with --save-taxonomy flag")
        return None
    
    with open(taxonomy_file, 'r') as f:
        taxonomy_data = json.load(f)
    
    taxonomy_train = taxonomy_data.get('train', [])
    taxonomy_val = taxonomy_data.get('val', [])
    
    input_length = train_data.shape[1]
    
    # Create and train hierarchical classifier
    classifier = HierarchicalClassifier(
        input_length=input_length,
        architecture=args.architecture
    )
    
    results = classifier.train_all_levels(
        train_data, None, val_data, None,
        taxonomy_train, taxonomy_val,
        epochs=args.epochs // 2,
        batch_size=args.batch_size
    )
    
    # Save
    output_path = Path(args.output_dir) / 'hierarchical'
    classifier.save(str(output_path))
    
    print(f"\nHierarchical classifier saved to {output_path}")
    
    return results


def train_hybrid(args):
    """Train hybrid CNN + k-mer model"""
    print("\n" + "="*70)
    print("TRAINING HYBRID CNN + K-MER MODEL")
    print("="*70)
    
    set_seeds(args.seed)
    
    # Load data
    train_data, val_data, test_data, train_labels, val_labels, test_labels, metadata = \
        load_dataset(args.data_dir)
    
    # Check for raw sequences
    sequences_file = Path(args.data_dir) / 'sequences.json'
    if not sequences_file.exists():
        print("ERROR: Hybrid model requires raw sequences.")
        print(f"Expected file: {sequences_file}")
        print("\nPlease run prepare_dataset.py with --save-sequences flag")
        return None
    
    with open(sequences_file, 'r') as f:
        sequences_data = json.load(f)
    
    sequences_train = sequences_data.get('train', [])
    sequences_val = sequences_data.get('val', [])
    
    num_classes = metadata['num_classes']
    input_length = train_data.shape[1]
    
    train_labels_cat = to_categorical(train_labels, num_classes)
    val_labels_cat = to_categorical(val_labels, num_classes)
    
    # Extract k-mer features
    print(f"\nExtracting {args.kmer_k}-mer features...")
    extractor = KmerExtractor(k=args.kmer_k)
    
    X_kmer_train = extractor.extract_batch(sequences_train)
    X_kmer_val = extractor.extract_batch(sequences_val)
    
    X_kmer_train = extractor.fit_transform(X_kmer_train)
    X_kmer_val = extractor.transform(X_kmer_val)
    
    print(f"  K-mer feature dimension: {X_kmer_train.shape[1]}")
    
    # Create and train hybrid model
    model = HybridCNNKmerModel(
        input_length=input_length,
        num_classes=num_classes,
        kmer_dim=X_kmer_train.shape[1]
    )
    
    print("\nModel architecture:")
    model.summary()
    
    print("\nTraining...")
    history = model.train(
        train_data, X_kmer_train, train_labels_cat,
        val_data, X_kmer_val, val_labels_cat,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Evaluate
    test_labels_cat = to_categorical(test_labels, num_classes)
    
    # Load test sequences
    sequences_test = sequences_data.get('test', [])
    X_kmer_test = extractor.extract_batch(sequences_test)
    X_kmer_test = extractor.transform(X_kmer_test)
    
    metrics = model.evaluate(test_data, X_kmer_test, test_labels_cat)
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Top-3 Accuracy: {metrics['top3_accuracy']:.4f}")
    
    # Save
    output_path = Path(args.output_dir) / 'hybrid'
    output_path.mkdir(parents=True, exist_ok=True)
    
    model.save(str(output_path / 'hybrid_model.h5'))
    extractor.save(str(output_path / 'kmer_extractor.json'))
    
    results = {
        'model': 'hybrid_cnn_kmer',
        'accuracy': float(metrics['accuracy']),
        'top3_accuracy': float(metrics['top3_accuracy']),
        'kmer_k': args.kmer_k
    }
    
    with open(output_path / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nHybrid model saved to {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Improved Training for Deep-Sea eDNA Classifier',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default='data/processed',
                       help='Directory with processed data')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for models and results')
    
    # General training arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    
    # Model architecture
    parser.add_argument('--architecture', type=str, default='attention_resnet',
                       choices=['attention_resnet', 'multiscale', 'deep_resnet'],
                       help='CNN architecture')
    
    # Training mode
    parser.add_argument('--train-improved', action='store_true',
                       help='Train improved CNN classifier')
    parser.add_argument('--train-ensemble', action='store_true',
                       help='Train ensemble of models')
    parser.add_argument('--train-hierarchical', action='store_true',
                       help='Train hierarchical classifier')
    parser.add_argument('--train-hybrid', action='store_true',
                       help='Train hybrid CNN + k-mer model')
    parser.add_argument('--train-all', action='store_true',
                       help='Train all model types')
    
    # Improvement options
    parser.add_argument('--use-class-weights', action='store_true', default=True,
                       help='Use balanced class weights')
    parser.add_argument('--no-class-weights', action='store_false', dest='use_class_weights',
                       help='Disable class weights')
    parser.add_argument('--augment', action='store_true',
                       help='Apply data augmentation')
    parser.add_argument('--augment-factor', type=int, default=2,
                       help='Augmentation factor (multiples of original data)')
    parser.add_argument('--use-mixup', action='store_true',
                       help='Use mixup augmentation')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                       help='Label smoothing factor')
    parser.add_argument('--save-best', action='store_true', default=True,
                       help='Save best model checkpoint')
    
    # Ensemble options
    parser.add_argument('--n-ensemble-models', type=int, default=5,
                       help='Number of models in ensemble')
    
    # Hybrid model options
    parser.add_argument('--kmer-k', type=int, default=4,
                       help='K-mer size for hybrid model')
    
    args = parser.parse_args()
    
    # Print configuration
    print("="*70)
    print("IMPROVED eDNA CLASSIFIER TRAINING")
    print("="*70)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Architecture: {args.architecture}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Class weights: {args.use_class_weights}")
    print(f"Data augmentation: {args.augment}")
    
    # Train requested models
    results = {}
    
    if args.train_all:
        args.train_improved = True
        args.train_ensemble = True
    
    if args.train_improved:
        results['improved'] = train_improved_classifier(args)
    
    if args.train_ensemble:
        results['ensemble'] = train_ensemble_models(args)
    
    if args.train_hierarchical:
        results['hierarchical'] = train_hierarchical(args)
    
    if args.train_hybrid:
        results['hybrid'] = train_hybrid(args)
    
    if not any([args.train_improved, args.train_ensemble, 
                args.train_hierarchical, args.train_hybrid, args.train_all]):
        print("\nNo training mode specified. Use one of:")
        print("  --train-improved    Train improved CNN classifier")
        print("  --train-ensemble    Train ensemble of models")
        print("  --train-hierarchical Train hierarchical classifier")
        print("  --train-hybrid      Train hybrid CNN + k-mer model")
        print("  --train-all         Train all model types")
        parser.print_help()
        return
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*70)
    
    for model_type, result in results.items():
        if result:
            print(f"\n{model_type.upper()}:")
            if 'accuracy' in result:
                print(f"  Accuracy: {result['accuracy']:.4f}")
            if 'ensemble_accuracy' in result:
                print(f"  Ensemble Accuracy: {result['ensemble_accuracy']:.4f}")
            if 'f1_macro' in result:
                print(f"  F1 (macro): {result['f1_macro']:.4f}")
    
    print("\n" + "="*70)
    print(f"All results saved to {args.output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
