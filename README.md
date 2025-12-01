# 🧬 Deep-Sea eDNA Classifier

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.13+](https://img.shields.io/badge/tensorflow-2.13+-orange.svg)](https://tensorflow.org/)

A deep learning framework for taxonomic classification of environmental DNA (eDNA) sequences from deep-sea environments. Achieves **96.97% accuracy** on 129 phylum-level taxa using a hybrid CNN architecture with attention mechanisms and k-mer frequency features.

---

## 📊 Performance Highlights

| Model | Accuracy | Top-3 Acc | Macro F1 | Size |
|-------|----------|-----------|----------|------|
| Baseline CNN | 87.80% | 95.94% | 0.87 | 9.02 MB |
| Improved CNN (Attention-ResNet) | 91.49% | 96.69% | 0.82 | 3.44 MB |
| Ensemble (3 models) | 95.72% | 98.10% | 0.88 | 10.3 MB |
| **Hybrid CNN + K-mer** | **96.97%** | **98.75%** | **0.93** | **3.19 MB** |

### Key Improvements Over Baseline
- ✅ **+9.17%** accuracy improvement
- ✅ **+2.81%** top-3 accuracy improvement  
- ✅ **-65%** model size reduction
- ✅ **89 classes** with F1 > 0.9 (vs 60 baseline)

---

## 🏗️ Architecture

### Hybrid CNN + K-mer Model

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT SEQUENCE (500 bp)                   │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│   SEQUENCE BRANCH       │     │    K-MER BRANCH         │
│                         │     │                         │
│  Conv1D (64, k=7)       │     │  4-mer frequencies      │
│  BatchNorm + ReLU       │     │  (256 features)         │
│  MaxPool1D              │     │                         │
│         ▼               │     │  Dense(128) + BN        │
│  ResidualBlock (64)     │     │  Dropout(0.3)           │
│  ResidualBlock (128)    │     │  Dense(64)              │
│  ResidualBlock (256)    │     │                         │
│         ▼               │     │                         │
│  ChannelAttention       │     │                         │
│  SelfAttention          │     │                         │
│  GlobalAvgPool → (256)  │     │         → (64)          │
└─────────────────────────┘     └─────────────────────────┘
              │                               │
              └───────────┬───────────────────┘
                          ▼
              ┌─────────────────────────┐
              │   CONCATENATE (320)     │
              │   Dense(256) + Dropout  │
              │   Dense(129) + Softmax  │
              └─────────────────────────┘
                          │
                          ▼
              ┌─────────────────────────┐
              │  OUTPUT: 129 classes    │
              └─────────────────────────┘
```

---

deep-sea-edna-classifier/
│
├── 📂 models/
│   ├── __init__.py
│   ├── ensemble_classifier.py
│   ├── hierarchical_classifier.py
│   ├── improved_cnn_classifier.py
│   ├── kmer_features.py
│
├── 📂 results_improved/
│   ├── abundance.png
│   ├── best_improved_model.h5
│   ├── class_distribution.png
│   ├── class_performance.png
│   ├── confusion_matrix.png
│   ├── final_results.json
│   ├── hybrid_model.h5
│   ├── improved_cnn_classifier.h5
│   ├── model_comparison.csv
│   ├── training_history.png
│
├── 📂 scripts/
│   ├── prepare_dataset.py
│   ├── train_improved.py
│
├── 📄 README.md
├── 📄 edna_improved_technical_report.pdf

```

---

## 📥 Dataset

### ⚠️ Dataset Not Included

The training dataset is **not included** in this repository due to size constraints. You need to download it separately.

### Download Instructions

#### 1. SILVA Database (Primary)

Download SILVA 138.1 SSURef NR99 from the official website:

```bash
# Download SILVA database (~2.5 GB compressed)
wget https://www.arb-silva.de/fileadmin/silva_databases/release_138.1/Exports/SILVA_138.1_SSURef_NR99_tax_silva.fasta.gz

# Extract
gunzip SILVA_138.1_SSURef_NR99_tax_silva.fasta.gz

# Move to data directory
mkdir -p data/raw
mv SILVA_138.1_SSURef_NR99_tax_silva.fasta data/raw/
```

#### 2. NCBI rRNA Databases (Optional - for improved model)

Download from NCBI using Entrez:

```bash
# Install Entrez Direct
sh -c "$(curl -fsSL ftp://ftp.ncbi.nlm.nih.gov/entrez/entrezdirect/install-edirect.sh)"

# Download 16S rRNA (Prokaryotes)
esearch -db nucleotide -query "16S ribosomal RNA[Title] AND (bacteria[filter] OR archaea[filter])" | \
  efetch -format fasta > data/raw/16S_ribosomal_RNA.fasta

# Download 18S rRNA (Eukaryotes)
esearch -db nucleotide -query "18S small subunit ribosomal RNA[Title] AND eukaryota[filter]" | \
  efetch -format fasta > data/raw/SSU_eukaryote_rRNA.fasta
```

#### 3. Combined Dataset

After downloading, combine the databases:

```bash
python scripts/preprocess.py \
    --silva data/raw/SILVA_138.1_SSURef_NR99_tax_silva.fasta \
    --ncbi data/raw/ \
    --output data/processed/ \
    --min-samples 10 \
    --max-samples 350
```

### Dataset Statistics

| Property | Value |
|----------|-------|
| Total Classes | 129 phyla |
| Min samples/class | 7 |
| Max samples/class | 350 |
| Median samples/class | 34 |
| Sequence Length | 500 bp (fixed) |
| Train/Val/Test Split | 70%/15%/15% |

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Bhaveshtambolii/deep-sea-edna-classifier.git
cd deep-sea-edna-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
tensorflow>=2.13.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
biopython>=1.81
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
h5py>=3.9.0
```

### Training

#### Using Notebooks (Recommended for first-time users)

1. Open `notebooks/03_improved_model_training.ipynb` in Google Colab or Jupyter
2. Upload your dataset to Google Drive
3. Run all cells

#### Using Scripts

```bash
# Train baseline model
python scripts/train.py \
    --model baseline \
    --data data/processed/ \
    --epochs 50 \
    --output models/baseline_cnn.h5

# Train improved model (Attention-ResNet)
python scripts/train.py \
    --model improved \
    --data data/processed/ \
    --epochs 100 \
    --augment \
    --output models/improved_cnn.h5

# Train hybrid model (best performance)
python scripts/train.py \
    --model hybrid \
    --data data/processed/ \
    --epochs 100 \
    --augment \
    --kmer 4 \
    --output models/hybrid_cnn_kmer.h5
```

### Prediction

```bash
# Predict on new sequences
python scripts/predict.py \
    --input your_sequences.fasta \
    --model models/hybrid_cnn_kmer.h5 \
    --output predictions.json \
    --batch-size 1000
```

---

## 📖 Usage Examples

### Python API

```python
import tensorflow as tf
import numpy as np
from Bio import SeqIO

# Load model
model = tf.keras.models.load_model('models/hybrid_cnn_kmer.h5')

# Encode sequence (one-hot)
def encode_sequence(seq, length=500):
    mapping = {'A': [1,0,0,0,0], 'C': [0,1,0,0,0], 
               'G': [0,0,1,0,0], 'T': [0,0,0,1,0], 'N': [0,0,0,0,1]}
    encoded = np.zeros((length, 5))
    for i, base in enumerate(seq[:length].upper()):
        encoded[i] = mapping.get(base, [0,0,0,0,1])
    return encoded

# Extract k-mer features
def extract_kmers(seq, k=4):
    from itertools import product
    kmers = [''.join(p) for p in product('ACGT', repeat=k)]
    kmer_to_idx = {kmer: i for i, kmer in enumerate(kmers)}
    counts = np.zeros(len(kmers))
    seq = seq.upper()
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        if kmer in kmer_to_idx:
            counts[kmer_to_idx[kmer]] += 1
    return counts / max(counts.sum(), 1)

# Predict
sequence = "ATCGATCGATCG..."  # Your DNA sequence
seq_encoded = encode_sequence(sequence)[np.newaxis, ...]
kmer_features = extract_kmers(sequence)[np.newaxis, ...]

prediction = model.predict([seq_encoded, kmer_features])
predicted_class = np.argmax(prediction)
confidence = prediction[0, predicted_class]

print(f"Predicted class: {predicted_class}")
print(f"Confidence: {confidence:.2%}")
```

---

## 📈 Results

### Training History
![Training History](results/training_history.png)

### Confusion Matrix
![Confusion Matrix](results/confusion_matrix.png)

### Per-Class Performance
![Class Performance](results/class_performance.png)

---

## 🧪 Comparison with Other Tools

| Tool | Accuracy | Speed | Notes |
|------|----------|-------|-------|
| BLASTN | ~85% | Slow | Database-dependent |
| Kraken2 | ~88% | Very fast | K-mer only |
| QIIME2 | ~82% | Moderate | Requires tuning |
| RDP Classifier | ~80% | Fast | Prokaryotes only |
| **Ours (Hybrid)** | **96.97%** | **718 seq/s** | Requires GPU |

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{tamboli2025deepsea,
  title={Deep-Sea eDNA Taxonomic Classification Using Deep Learning 
         on Combined SILVA + NCBI Ribosomal RNA Databases},
  author={Tamboli, Bhavesh},
  journal={bioRxiv},
  year={2025}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

**Bhavesh Tamboli**
- GitHub: [@Bhaveshtambolii](https://github.com/Bhaveshtambolii)

---

## 🙏 Acknowledgments

- [SILVA Database](https://www.arb-silva.de/) for ribosomal RNA sequences
- [NCBI](https://www.ncbi.nlm.nih.gov/) for curated rRNA databases
- TensorFlow team for the deep learning framework
- Google Colab for GPU resources

---

<p align="center">
  <b>⭐ Star this repository if you find it useful! ⭐</b>
</p>
