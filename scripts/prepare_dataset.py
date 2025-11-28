#!/usr/bin/env python3
"""
Improved Dataset Preparation Script for Deep-Sea eDNA Classifier
Includes support for multiple databases, augmentation, and balanced sampling
"""

import argparse
import json
import gzip
import random
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
import numpy as np

try:
    from Bio import SeqIO
    from Bio.Seq import Seq
except ImportError:
    print("BioPython not installed. Run: pip install biopython")
    exit(1)

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.augmentation import DNAAugmenter, encode_sequence


def parse_pr2_taxonomy(header: str) -> Dict[str, str]:
    """Parse PR2 taxonomy from FASTA header"""
    parts = header.split(';')
    taxonomy = {
        'domain': 'Eukaryota',
        'kingdom': parts[0] if len(parts) > 0 else 'Unknown',
        'phylum': parts[1] if len(parts) > 1 else 'Unknown',
        'class': parts[2] if len(parts) > 2 else 'Unknown',
        'order': parts[3] if len(parts) > 3 else 'Unknown',
        'family': parts[4] if len(parts) > 4 else 'Unknown',
        'genus': parts[5] if len(parts) > 5 else 'Unknown',
        'species': parts[6] if len(parts) > 6 else 'Unknown'
    }
    return taxonomy


def parse_silva_taxonomy(header: str) -> Dict[str, str]:
    """Parse SILVA taxonomy from FASTA header"""
    parts = header.split(';')
    taxonomy = {
        'domain': parts[0] if len(parts) > 0 else 'Unknown',
        'phylum': parts[1] if len(parts) > 1 else 'Unknown',
        'class': parts[2] if len(parts) > 2 else 'Unknown',
        'order': parts[3] if len(parts) > 3 else 'Unknown',
        'family': parts[4] if len(parts) > 4 else 'Unknown',
        'genus': parts[5] if len(parts) > 5 else 'Unknown',
        'species': parts[6] if len(parts) > 6 else 'Unknown'
    }
    return taxonomy


def load_fasta(filepath: str, max_sequences: Optional[int] = None) -> List[Tuple[str, str, str]]:
    """
    Load sequences from FASTA file
    
    Args:
        filepath: Path to FASTA file (can be gzipped)
        max_sequences: Maximum number of sequences to load
        
    Returns:
        List of (id, sequence, description) tuples
    """
    sequences = []
    
    # Handle gzipped files
    if filepath.endswith('.gz'):
        handle = gzip.open(filepath, 'rt')
    else:
        handle = open(filepath, 'r')
    
    try:
        for i, record in enumerate(SeqIO.parse(handle, 'fasta')):
            if max_sequences and i >= max_sequences:
                break
            sequences.append((record.id, str(record.seq), record.description))
    finally:
        handle.close()
    
    return sequences


def filter_sequences(
    sequences: List[Tuple[str, str, str]],
    min_length: int = 200,
    max_length: int = 2000,
    max_ambiguous: float = 0.05
) -> List[Tuple[str, str, str]]:
    """
    Filter sequences by quality criteria
    
    Args:
        sequences: List of (id, sequence, description) tuples
        min_length: Minimum sequence length
        max_length: Maximum sequence length
        max_ambiguous: Maximum fraction of ambiguous bases (N)
        
    Returns:
        Filtered sequences
    """
    filtered = []
    
    for seq_id, seq, desc in sequences:
        # Check length
        if len(seq) < min_length or len(seq) > max_length:
            continue
        
        # Check ambiguous bases
        n_count = seq.upper().count('N')
        if n_count / len(seq) > max_ambiguous:
            continue
        
        filtered.append((seq_id, seq, desc))
    
    print(f"Filtered {len(sequences)} -> {len(filtered)} sequences")
    return filtered


def extract_taxonomy_labels(
    sequences: List[Tuple[str, str, str]],
    level: str = 'family',
    parser: str = 'pr2'
) -> Tuple[List[str], List[str]]:
    """
    Extract taxonomy labels at specified level
    
    Args:
        sequences: List of (id, sequence, description) tuples
        level: Taxonomy level to extract
        parser: Parser type ('pr2' or 'silva')
        
    Returns:
        Tuple of (sequences, labels)
    """
    seqs = []
    labels = []
    
    parse_fn = parse_pr2_taxonomy if parser == 'pr2' else parse_silva_taxonomy
    
    for seq_id, seq, desc in sequences:
        try:
            tax = parse_fn(desc)
            label = tax.get(level, 'Unknown')
            
            if label and label != 'Unknown':
                seqs.append(seq)
                labels.append(label)
        except Exception as e:
            continue
    
    return seqs, labels


def balance_dataset(
    sequences: List[str],
    labels: List[str],
    min_samples: int = 10,
    max_samples: int = 500
) -> Tuple[List[str], List[str]]:
    """
    Balance dataset by over/under-sampling classes
    
    Args:
        sequences: List of sequences
        labels: List of labels
        min_samples: Minimum samples per class (remove if fewer)
        max_samples: Maximum samples per class (downsample if more)
        
    Returns:
        Balanced (sequences, labels)
    """
    # Group by label
    label_groups = defaultdict(list)
    for seq, label in zip(sequences, labels):
        label_groups[label].append(seq)
    
    balanced_seqs = []
    balanced_labels = []
    
    for label, seqs in label_groups.items():
        if len(seqs) < min_samples:
            print(f"  Removing {label}: only {len(seqs)} samples")
            continue
        
        if len(seqs) > max_samples:
            # Downsample
            seqs = random.sample(seqs, max_samples)
        
        balanced_seqs.extend(seqs)
        balanced_labels.extend([label] * len(seqs))
    
    return balanced_seqs, balanced_labels


def encode_sequences(
    sequences: List[str],
    max_length: int = 500,
    padding: str = 'post'
) -> np.ndarray:
    """
    One-hot encode DNA sequences
    
    Args:
        sequences: List of DNA sequences
        max_length: Maximum sequence length (pad/truncate)
        padding: Padding position ('pre' or 'post')
        
    Returns:
        Encoded sequences [n_samples, max_length, 5]
    """
    encoded = []
    
    for seq in sequences:
        enc = encode_sequence(seq, max_length)
        encoded.append(enc)
    
    return np.array(encoded, dtype=np.float32)


def encode_labels(labels: List[str]) -> Tuple[np.ndarray, Dict[str, int], Dict[int, str]]:
    """
    Encode string labels to integers
    
    Args:
        labels: List of string labels
        
    Returns:
        Tuple of (encoded_labels, label_to_idx, idx_to_label)
    """
    unique_labels = sorted(set(labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    
    encoded = np.array([label_to_idx[label] for label in labels], dtype=np.int32)
    
    return encoded, label_to_idx, idx_to_label


def split_dataset(
    sequences: List[str],
    labels: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    stratify: bool = True,
    seed: int = 42
) -> Dict[str, Tuple[List[str], np.ndarray]]:
    """
    Split dataset into train/val/test
    
    Args:
        sequences: List of sequences
        labels: Encoded labels
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        stratify: Stratified split by label
        seed: Random seed
        
    Returns:
        Dictionary with train/val/test splits
    """
    random.seed(seed)
    np.random.seed(seed)
    
    n_samples = len(sequences)
    indices = list(range(n_samples))
    
    if stratify:
        # Stratified split
        from sklearn.model_selection import train_test_split
        
        # First split: train vs (val + test)
        train_idx, temp_idx = train_test_split(
            indices, train_size=train_ratio, stratify=labels, random_state=seed
        )
        
        # Second split: val vs test
        val_size = val_ratio / (1 - train_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx, train_size=val_size, stratify=labels[temp_idx], random_state=seed
        )
    else:
        random.shuffle(indices)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
    
    return {
        'train': ([sequences[i] for i in train_idx], labels[train_idx]),
        'val': ([sequences[i] for i in val_idx], labels[val_idx]),
        'test': ([sequences[i] for i in test_idx], labels[test_idx])
    }


def augment_training_data(
    sequences: List[str],
    labels: np.ndarray,
    augment_factor: int = 2,
    seed: int = 42
) -> Tuple[List[str], np.ndarray]:
    """
    Augment training data with DNA-specific augmentations
    
    Args:
        sequences: List of sequences
        labels: Encoded labels
        augment_factor: Multiplication factor
        seed: Random seed
        
    Returns:
        Augmented (sequences, labels)
    """
    augmenter = DNAAugmenter(seed=seed)
    
    augmented_seqs = list(sequences)
    augmented_labels = list(labels)
    
    for _ in range(augment_factor - 1):
        for seq, label in zip(sequences, labels):
            # Generate augmented sequence
            aug_seq = augmenter.augment(seq, p=0.5)
            augmented_seqs.append(aug_seq)
            augmented_labels.append(label)
    
    return augmented_seqs, np.array(augmented_labels)


def prepare_dataset(args):
    """Main dataset preparation function"""
    print("="*70)
    print("PREPARING eDNA DATASET")
    print("="*70)
    
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Load sequences
    print(f"\nLoading sequences from {args.input}...")
    sequences = load_fasta(args.input, max_sequences=args.max_sequences)
    print(f"  Loaded {len(sequences)} sequences")
    
    # Filter by quality
    print("\nFiltering sequences...")
    sequences = filter_sequences(
        sequences,
        min_length=args.min_length,
        max_length=args.max_length,
        max_ambiguous=args.max_ambiguous
    )
    
    # Extract taxonomy labels
    print(f"\nExtracting {args.taxonomy_level} labels...")
    seqs, labels = extract_taxonomy_labels(
        sequences,
        level=args.taxonomy_level,
        parser=args.parser
    )
    print(f"  Found {len(set(labels))} unique {args.taxonomy_level} labels")
    
    # Balance dataset
    print("\nBalancing dataset...")
    seqs, labels = balance_dataset(
        seqs, labels,
        min_samples=args.min_samples,
        max_samples=args.max_samples
    )
    print(f"  Final: {len(seqs)} sequences, {len(set(labels))} classes")
    
    # Encode labels
    encoded_labels, label_to_idx, idx_to_label = encode_labels(labels)
    
    # Split dataset
    print("\nSplitting dataset...")
    splits = split_dataset(
        seqs, encoded_labels,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        stratify=True,
        seed=args.seed
    )
    
    print(f"  Train: {len(splits['train'][0])} samples")
    print(f"  Val: {len(splits['val'][0])} samples")
    print(f"  Test: {len(splits['test'][0])} samples")
    
    # Augment training data
    if args.augment:
        print(f"\nAugmenting training data (factor={args.augment_factor})...")
        train_seqs, train_labels = augment_training_data(
            splits['train'][0],
            splits['train'][1],
            augment_factor=args.augment_factor,
            seed=args.seed
        )
        splits['train'] = (train_seqs, train_labels)
        print(f"  Augmented training: {len(train_seqs)} samples")
    
    # Encode sequences
    print(f"\nEncoding sequences (max_length={args.sequence_length})...")
    encoded_data = {}
    for split_name, (split_seqs, split_labels) in splits.items():
        encoded_data[split_name] = {
            'data': encode_sequences(split_seqs, max_length=args.sequence_length),
            'labels': split_labels,
            'sequences': split_seqs
        }
        print(f"  {split_name}: {encoded_data[split_name]['data'].shape}")
    
    # Save data
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving to {output_dir}...")
    
    # Save encoded data
    for split_name, data in encoded_data.items():
        np.save(output_dir / f'{split_name}_data.npy', data['data'])
        np.save(output_dir / f'{split_name}_labels.npy', data['labels'])
    
    # Save metadata
    metadata = {
        'num_classes': len(label_to_idx),
        'sequence_length': args.sequence_length,
        'encoding_dim': 5,
        'taxonomy_level': args.taxonomy_level,
        'label_mapping': {str(v): k for k, v in label_to_idx.items()},
        'class_counts': dict(Counter(encoded_labels)),
        'train_samples': len(splits['train'][0]),
        'val_samples': len(splits['val'][0]),
        'test_samples': len(splits['test'][0]),
        'augmented': args.augment
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save raw sequences (for k-mer extraction)
    if args.save_sequences:
        sequences_data = {
            'train': splits['train'][0],
            'val': splits['val'][0],
            'test': splits['test'][0]
        }
        with open(output_dir / 'sequences.json', 'w') as f:
            json.dump(sequences_data, f)
        print("  Saved raw sequences")
    
    # Save taxonomy labels (for hierarchical classification)
    if args.save_taxonomy:
        # Get full taxonomy strings
        taxonomy_data = {}
        for split_name, (split_seqs, split_labels) in splits.items():
            tax_labels = []
            for label_idx in split_labels:
                label_name = idx_to_label[label_idx]
                tax_labels.append(label_name)
            taxonomy_data[split_name] = tax_labels
        
        with open(output_dir / 'taxonomy_labels.json', 'w') as f:
            json.dump(taxonomy_data, f)
        print("  Saved taxonomy labels")
    
    # Print class distribution
    print("\nClass distribution (top 20):")
    class_counts = Counter(encoded_labels)
    for i, (label_idx, count) in enumerate(class_counts.most_common(20)):
        label_name = idx_to_label[label_idx]
        print(f"  {i+1:2d}. {label_name:40s}: {count:5d}")
    
    print("\n" + "="*70)
    print("DATASET PREPARATION COMPLETE")
    print("="*70)
    print(f"Output directory: {output_dir}")
    print(f"Total classes: {len(label_to_idx)}")
    print(f"Total samples: {sum(len(s[0]) for s in splits.values())}")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare eDNA dataset for deep learning',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output
    parser.add_argument('--input', type=str, required=True,
                       help='Input FASTA file (can be gzipped)')
    parser.add_argument('--output', type=str, default='data/processed',
                       help='Output directory')
    
    # Sequence filtering
    parser.add_argument('--min-length', type=int, default=200,
                       help='Minimum sequence length')
    parser.add_argument('--max-length', type=int, default=2000,
                       help='Maximum sequence length')
    parser.add_argument('--max-ambiguous', type=float, default=0.05,
                       help='Maximum fraction of ambiguous bases')
    parser.add_argument('--max-sequences', type=int, default=None,
                       help='Maximum number of sequences to load')
    
    # Taxonomy
    parser.add_argument('--taxonomy-level', type=str, default='family',
                       choices=['domain', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus'],
                       help='Taxonomy level for classification')
    parser.add_argument('--parser', type=str, default='pr2',
                       choices=['pr2', 'silva'],
                       help='Taxonomy parser')
    
    # Balancing
    parser.add_argument('--min-samples', type=int, default=10,
                       help='Minimum samples per class')
    parser.add_argument('--max-samples', type=int, default=500,
                       help='Maximum samples per class')
    
    # Encoding
    parser.add_argument('--sequence-length', type=int, default=500,
                       help='Target sequence length (pad/truncate)')
    
    # Splits
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='Validation set ratio')
    
    # Augmentation
    parser.add_argument('--augment', action='store_true',
                       help='Augment training data')
    parser.add_argument('--augment-factor', type=int, default=2,
                       help='Augmentation factor')
    
    # Additional outputs
    parser.add_argument('--save-sequences', action='store_true',
                       help='Save raw sequences (for k-mer extraction)')
    parser.add_argument('--save-taxonomy', action='store_true',
                       help='Save full taxonomy labels (for hierarchical classification)')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    prepare_dataset(args)


if __name__ == '__main__':
    main()
