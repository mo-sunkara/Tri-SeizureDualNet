# Tri-SeizureDualNet

A deep learning model for seizure detection using dual-stream neural network architecture that processes both EEG and fMRI data.
 

## Features

- Dual-stream neural network architecture
- Processes both EEG and fMRI data simultaneously
- Memory-efficient data loading for large datasets
- Balanced batch generation for handling imbalanced datasets
- Mixed precision training for improved performance

## Requirements

- Python 3.7+
- TensorFlow 2.11.0+
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/Tri-SeizureDualNet.git
cd Tri-SeizureDualNet
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Model Architecture

The model consists of:
- Triple-stream feature extractor for EEG data
- HBO feature selector for fMRI data
- Dual parallel attention transformer
- LSTM layers for temporal processing

## Usage

```python
from tri_seizure_dualnet import TriSeizureDualNet

# Create model instance
model = TriSeizureDualNet()

# Train model
model.fit([X_eeg, X_fmri], y)

# Make predictions
predictions = model.predict([X_eeg_test, X_fmri_test])
```