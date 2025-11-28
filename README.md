# Deep-Sea eDNA Classifier - Improved Version

A comprehensive deep learning pipeline for classifying environmental DNA (eDNA) sequences from deep-sea samples. This improved version includes residual networks, attention mechanisms, ensemble methods, hierarchical classification, and k-mer features.

## 🚀 Key Improvements

| Feature | Description | Expected Gain |
|---------|-------------|---------------|
| **Class Weights** | Balanced handling of rare species | +1-2% accuracy |
| **Data Augmentation** | DNA-specific mutations, reverse complement | +1-2% accuracy |
| **Residual + Attention** | Modern architecture with skip connections | +2-3% accuracy |
| **Ensemble Methods** | Multiple models with weighted voting | +1-2% accuracy |
| **Hierarchical Classification** | Domain → Phylum → Class cascading | +3-5% accuracy |
| **K-mer Features** | Hybrid CNN + frequency features | +1-2% accuracy |

## 📁 Project Structure

```
edna_improved/
├── models/
│   ├── improved_cnn_classifier.py    # ResNet + Attention architecture
│   ├── ensemble_classifier.py         # Ensemble methods
│   ├── hierarchical_classifier.py     # Multi-level taxonomy
│   └── kmer_features.py               # K-mer extraction & hybrid model
├── utils/
│   ├── augmentation.py                # DNA-specific augmentation
│   └── evaluation.py                  # Metrics & visualization
├── scripts/
│   ├── prepare_dataset.py             # Data preparation
│   └── train_improved.py              # Training pipeline
├── data/                              # Data directory
├── results/                           # Output directory
├── requirements.txt
└── README.md
```

## 🛠️ Installation

```bash
# Clone or copy this directory
cd edna_improved

# Install dependencies
pip install -r requirements.txt
```

## 📊 Quick Start

### 1. Prepare Dataset

```bash
python scripts/prepare_dataset.py \
    --input data/raw/pr2_version_5.0.0_SSU_mothur.fasta \
    --output data/processed \
    --taxonomy-level family \
    --min-samples 10 \
    --max-samples 500 \
    --sequence-length 500 \
    --augment \
    --save-sequences \
    --save-taxonomy
```

### 2. Train Improved Classifier

```bash
# Train with all improvements
python scripts/train_improved.py \
    --train-improved \
    --data-dir data/processed \
    --output-dir results \
    --architecture attention_resnet \
    --use-class-weights \
    --augment \
    --epochs 100
```

### 3. Train Ensemble

```bash
python scripts/train_improved.py \
    --train-ensemble \
    --data-dir data/processed \
    --output-dir results \
    --n-ensemble-models 5 \
    --epochs 50
```

### 4. Train All Models

```bash
python scripts/train_improved.py \
    --train-all \
    --data-dir data/processed \
    --output-dir results \
    --epochs 100
```

## 🏗️ Model Architectures

### 1. Attention ResNet (Default)
```python
from models.improved_cnn_classifier import create_improved_model

model = create_improved_model(
    input_length=500,
    num_classes=126,
    architecture='attention_resnet'
)
```

Features:
- Residual blocks with skip connections
- Channel attention (Squeeze-and-Excitation)
- Self-attention for sequence dependencies
- Label smoothing & class weights

### 2. Multi-Scale CNN
```python
model = create_improved_model(
    input_length=500,
    num_classes=126,
    architecture='multiscale'
)
```

Features:
- Parallel convolutions at different scales (3, 5, 7, 11)
- Captures both local and global patterns
- Combined avg/max pooling

### 3. Deep ResNet
```python
model = create_improved_model(
    input_length=500,
    num_classes=126,
    architecture='deep_resnet'
)
```

Features:
- 11 residual blocks
- Up to 512 filters
- Best for very large datasets

## 🔧 Usage Examples

### Data Augmentation

```python
from utils.augmentation import DNAAugmenter

augmenter = DNAAugmenter(seed=42)

# Reverse complement (biologically equivalent)
rc = augmenter.reverse_complement(sequence)

# Random mutations (1% rate)
mutated = augmenter.random_mutation(sequence, mutation_rate=0.01)

# Generate augmented batch
batch = augmenter.generate_augmented_batch(sequence, n_augmentations=5)
```

### Ensemble Prediction

```python
from models.ensemble_classifier import EnsembleClassifier

ensemble = EnsembleClassifier(
    input_length=500,
    num_classes=126,
    n_models=5
)

# Train
ensemble.train(X_train, y_train, X_val, y_val, epochs=50)

# Predict with confidence
predictions, confidence, agreement = ensemble.predict_with_confidence(X_test)
```

### Hierarchical Classification

```python
from models.hierarchical_classifier import HierarchicalClassifier

classifier = HierarchicalClassifier(input_length=500)

# Train all levels
results = classifier.train_all_levels(
    X_train, None, X_val, None,
    taxonomy_train, taxonomy_val
)

# Predict with cascading
predictions = classifier.predict(X_test)
# Returns: domain, phylum, class with probabilities
```

### K-mer Hybrid Model

```python
from models.kmer_features import KmerExtractor, HybridCNNKmerModel

# Extract k-mer features
extractor = KmerExtractor(k=4)
kmer_features = extractor.extract_batch(sequences)

# Train hybrid model
model = HybridCNNKmerModel(
    input_length=500,
    num_classes=126,
    kmer_dim=256
)
model.train(X_seq, X_kmer, y, ...)
```

## 📈 Evaluation

```python
from utils.evaluation import ModelEvaluator, AbundanceEstimator

# Comprehensive evaluation
evaluator = ModelEvaluator(class_names=class_names)
report = evaluator.generate_report(
    y_true, y_pred, y_prob,
    output_dir='results/evaluation'
)

# Generates:
# - Confusion matrix
# - Per-class F1 scores
# - ROC curves
# - Error analysis

# Abundance estimation
abundance = AbundanceEstimator()
relative_abundance = abundance.estimate_from_probabilities(y_prob, class_names)
abundance.plot_abundance(top_n=20, save_path='results/abundance.png')
```

## ⚙️ Command Line Options

### prepare_dataset.py

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | Required | Input FASTA file |
| `--output` | data/processed | Output directory |
| `--taxonomy-level` | family | Classification level |
| `--min-samples` | 10 | Min samples per class |
| `--max-samples` | 500 | Max samples per class |
| `--sequence-length` | 500 | Target sequence length |
| `--augment` | False | Enable augmentation |
| `--save-sequences` | False | Save raw sequences |
| `--save-taxonomy` | False | Save full taxonomy |

### train_improved.py

| Option | Default | Description |
|--------|---------|-------------|
| `--train-improved` | False | Train improved CNN |
| `--train-ensemble` | False | Train ensemble |
| `--train-hierarchical` | False | Train hierarchical |
| `--train-hybrid` | False | Train hybrid model |
| `--train-all` | False | Train all models |
| `--architecture` | attention_resnet | Model architecture |
| `--epochs` | 100 | Training epochs |
| `--use-class-weights` | True | Balance class weights |
| `--augment` | False | Runtime augmentation |
| `--n-ensemble-models` | 5 | Number in ensemble |

## 🎯 Expected Results

With all improvements applied:

| Metric | Baseline | Improved |
|--------|----------|----------|
| Accuracy | 87.8% | ~93-95% |
| Top-3 Accuracy | 95.9% | ~98-99% |
| F1 (macro) | ~0.82 | ~0.90 |
| Weak classes (F1<0.7) | 13 | <5 |

## 📝 Citation

If you use this code, please cite:
```
Deep-Sea eDNA Classifier: An improved deep learning approach 
for taxonomic classification of environmental DNA sequences.
```

## 📄 License

MIT License - feel free to use and modify.
