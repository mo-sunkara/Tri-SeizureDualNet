import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
import requests
import tempfile
import gc
from tensorflow.keras.mixed_precision import set_global_policy
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import FastICA
import scipy.stats
import mne
import traceback
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation, MaxPooling2D,
    Dropout, Flatten, Dense, GaussianNoise, Reshape, concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow.metrics

# Configure TensorFlow GPU settings
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled")
    except RuntimeError as e:
        print(f"Error configuring GPU: {e}")

# Disable OneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging except errors

# Enable mixed precision training for memory efficiency
set_global_policy('mixed_float16')

# Memory-efficient data generator
class DataGenerator(tf.keras.utils.Sequence):
    """Memory-efficient data generator"""
    def __init__(self, X_eeg, X_fmri, y, batch_size=32, shuffle=True):
        self.X_eeg = X_eeg
        self.X_fmri = X_fmri
        self.y = y.reshape(-1, 1)  # Reshape labels to (n_samples, 1)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.y))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        """Number of batches per epoch"""
        return int(np.ceil(len(self.y) / self.batch_size))

    def __getitem__(self, idx):
        """Get batch at position idx"""
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.y))
        batch_indexes = self.indexes[start_idx:end_idx]

        # Get batch data
        batch_X_eeg = self.X_eeg[batch_indexes].astype(np.float32)
        batch_X_fmri = self.X_fmri[batch_indexes].astype(np.float32)
        batch_y = self.y[batch_indexes].astype(np.float32)

        return batch_X_eeg, batch_X_fmri, batch_y

    def on_epoch_end(self):
        """Shuffle indexes after each epoch if shuffle is set to True"""
        if self.shuffle:
            np.random.shuffle(self.indexes)

class BalancedBatchGenerator(tf.keras.utils.Sequence):
    """Memory-efficient balanced batch generator for large datasets"""
    def __init__(self, X_eeg, X_fmri, y, batch_size=32, shuffle=True):
        self.X_eeg = X_eeg
        self.X_fmri = X_fmri
        self.y = y.reshape(-1, 1)
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Split indices by class for balanced sampling
        self.class_indices = [np.where(self.y == i)[0] for i in np.unique(self.y)]
        self.n_classes = len(self.class_indices)
        self.samples_per_class = self.batch_size // self.n_classes
        self.max_samples = min(len(indices) for indices in self.class_indices)
        
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.max_samples / self.samples_per_class))

    def __getitem__(self, idx):
        batch_indices = []
        
        # Sample equally from each class
        for class_idx in range(self.n_classes):
            start_idx = idx * self.samples_per_class
            end_idx = min((idx + 1) * self.samples_per_class, len(self.class_indices[class_idx]))
            batch_indices.extend(self.class_indices[class_idx][start_idx:end_idx])
        
        # Get batch data
        batch_X_eeg = self.X_eeg[batch_indices].astype(np.float32)
        batch_X_fmri = self.X_fmri[batch_indices].astype(np.float32)
        batch_y = self.y[batch_indices].astype(np.float32)
        
        return [batch_X_eeg, batch_X_fmri], batch_y

    def on_epoch_end(self):
        if self.shuffle:
            for indices in self.class_indices:
                np.random.shuffle(indices)

class TransposeLayer(tf.keras.layers.Layer):
    def __init__(self, perm, **kwargs):
        super().__init__(**kwargs)
        self.perm = perm
    
    def call(self, inputs):
        return tf.transpose(inputs, perm=self.perm)

class TripleStreamFeatureExtractor(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.spatial_stream = None
        self.spectral_stream = None
        self.temporal_stream = None
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        self.dropout = tf.keras.layers.Dropout(0.5)
    
    def build(self, input_shape):
        self.spatial_stream = self._build_stream(3, 'spatial')
        self.spectral_stream = self._build_stream(5, 'spectral')
        self.temporal_stream = self._build_stream(40, 'temporal')
    
    def _build_stream(self, kernel_size, name):
        return tf.keras.Sequential([
            TransposeLayer(perm=[0, 2, 1], name=f"{name}_transpose_1"),
            tf.keras.layers.Conv1D(64, kernel_size, padding='same', activation='relu', name=f"{name}_conv1",
                                   kernel_regularizer=l2(0.001)),
            tf.keras.layers.BatchNormalization(name=f"{name}_bn1"),
            tf.keras.layers.Dropout(0.3),  
            tf.keras.layers.Conv1D(32, kernel_size, padding='same', activation='relu', name=f"{name}_conv2",
                                   kernel_regularizer=l2(0.001)),
            tf.keras.layers.BatchNormalization(name=f"{name}_bn2"),
            tf.keras.layers.Dropout(0.3),   
            TransposeLayer(perm=[0, 2, 1], name=f"{name}_transpose_2"),
        ])
    
    def call(self, inputs, training=None):
        # Process through each stream
        spatial_out = self.spatial_stream(inputs, training=training)
        spectral_out = self.spectral_stream(inputs, training=training)
        temporal_out = self.temporal_stream(inputs, training=training)
        
        # Concatenate the outputs
        combined = self.concat([spatial_out, spectral_out, temporal_out])
        return self.dropout(combined, training=training)

class HBOFeatureSelector(tf.keras.layers.Layer):
    def __init__(self, n_features=128, **kwargs):
        super().__init__(**kwargs)
        self.n_features = n_features
        self.dense = tf.keras.layers.Dense(n_features, activation='relu', kernel_regularizer=l2(0.001))
    
    def call(self, inputs):
        return self.dense(inputs)

class DualParallelAttentionTransformer(layers.Layer):
    def __init__(self, units=32, num_heads=4, dropout=0.3, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.dropout_rate = dropout

        # Enhanced with regularization
        self.self_attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=units//num_heads,
            kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.dense_proj = tf.keras.Sequential([
            layers.Dense(units*2, activation='relu', 
                        kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            layers.Dropout(dropout),
            layers.Dense(units, kernel_regularizer=tf.keras.regularizers.l2(0.01))
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout = layers.Dropout(dropout)

    def call(self, inputs):
        attn_output = self.self_attention(inputs, inputs)
        proj_input = self.layernorm1(inputs + self.dropout(attn_output))
        proj_output = self.dense_proj(proj_input)
        return self.layernorm2(proj_input + self.dropout(proj_output))

class TriSeizureDualNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.feature_extractor = TripleStreamFeatureExtractor()
        self.hbo_selector = HBOFeatureSelector(n_features=128)
        self.transformer = DualParallelAttentionTransformer()
        self.dropout = tf.keras.layers.Dropout(0.5)
    
    def call(self, inputs):
        x = self.feature_extractor(inputs)
        x = self.hbo_selector(x)
        x = self.transformer(x)
        return self.dropout(x)

# Stream EEG data from a URL
def stream_eeg_data(url):
    """Stream EEG data from URL and handle channel names"""
    log(f"Streaming data from {url}")
    
    # Create a temporary file
    temp_file_handle, temp_file_path = tempfile.mkstemp(suffix='.edf')
    os.close(temp_file_handle)
    
    try:
        # Download the file
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(temp_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        log(f"Saved streamed data to temporary file: {temp_file_path}")
        
        # Read EDF file with custom naming to avoid duplicates
        raw = mne.io.read_raw_edf(temp_file_path, preload=True, verbose=False)
        raw.rename_channels(lambda x: x + '-dup' if raw.ch_names.count(x) > 1 else x)
        
        return raw, temp_file_path
        
    except Exception as e:
        os.remove(temp_file_path)
        raise Exception(f"Error streaming data: {str(e)}")

# Load fMRI data
def load_fmri_data(directory):
    """
    Load and preprocess fMRI data
    """
    log("Loading fMRI data from directory: " + os.path.basename(directory))
    
    try:
        # Load connectivity matrix
        conn_file = os.path.join(directory, 'BNA_cont.mat')
        log(f"Loading file: {conn_file}")
        conn_data = loadmat(conn_file)
        
        # Find the actual variable name in the .mat file
        conn_vars = [k for k in conn_data.keys() if not k.startswith('__')]
        if not conn_vars:
            log("No valid variables found in BNA_cont.mat")
            return None
            
        conn = conn_data[conn_vars[0]]
        log(f"Found connectivity matrix with shape: {conn.shape}")
        
        # Load individual subject data
        all_data = []
        for i in range(1, 9):  # Load 8 subjects
            file_path = os.path.join(directory, f'x{i}.mat')
            log(f"Loading file: {file_path}")
            
            try:
                data = loadmat(file_path)
                # Find the actual variable name
                data_vars = [k for k in data.keys() if not k.startswith('__')]
                if not data_vars:
                    log(f"No valid variables found in {file_path}")
                    continue
                    
                subject_data = data[data_vars[0]]
                
                # Ensure correct shape and add channel dimension if needed
                if subject_data.ndim == 2:
                    subject_data = np.expand_dims(subject_data, axis=-1)
                
                log(f"Loaded subject data with shape: {subject_data.shape}")
                all_data.append(subject_data)
                
            except Exception as e:
                log(f"Error loading {file_path}: {str(e)}")
                continue
        
        if not all_data:
            log("No valid fMRI data could be loaded")
            return None
            
        # Stack all subjects
        fmri_data = np.stack(all_data)
        
        # Ensure we have the channel dimension
        if fmri_data.ndim == 3:
            fmri_data = np.expand_dims(fmri_data, axis=-1)
        
        log(f"Combined fMRI data shape: {fmri_data.shape}")
        return fmri_data
        
    except Exception as e:
        log(f"Error loading fMRI data: {str(e)}")
        traceback.print_exc()  # Print full traceback for debugging
        return None

def generate_synthetic_fmri(real_fmri_data):
    """
    Generate synthetic fMRI data using statistical properties of real data
    """
    mean = np.mean(real_fmri_data, axis=0)
    std = np.std(real_fmri_data, axis=0)
    
    # Generate synthetic samples
    n_synthetic = max(50 - len(real_fmri_data), 0)
    synthetic_data = []
    
    for _ in range(n_synthetic):
        # Generate base random data
        synthetic = np.random.normal(mean, std)
        
        # Apply spatial smoothing to maintain local correlations
        synthetic = gaussian_filter(synthetic, sigma=1.0)
        
        # Ensure same shape as real data
        if synthetic.ndim != real_fmri_data.ndim:
            synthetic = synthetic.reshape(real_fmri_data.shape[1:])
        
        synthetic_data.append(synthetic)
    
    if synthetic_data:
        synthetic_data = np.array(synthetic_data)
        # Ensure same scale as real data
        synthetic_data = (synthetic_data - np.mean(synthetic_data)) / np.std(synthetic_data)
        synthetic_data = synthetic_data * np.std(real_fmri_data) + np.mean(real_fmri_data)
    else:
        synthetic_data = np.empty((0,) + real_fmri_data.shape[1:])
    
    return synthetic_data

def get_seizure_info(base_url):
    """Get list of files with seizures and their timings"""
    try:
        # Get list of files with seizures
        seizure_files_url = f"{base_url}/RECORDS-WITH-SEIZURES"
        response = requests.get(seizure_files_url)
        response.raise_for_status()
        seizure_files = response.text.strip().split('\n')
        
        # Get subject info
        subject_info_url = f"{base_url}/SUBJECT-INFO"
        response = requests.get(subject_info_url)
        response.raise_for_status()
        subject_info = response.text.strip().split('\n')
        
        return seizure_files, subject_info
    except Exception as e:
        log(f"Error getting seizure info: {e}")
        return [], []

def augment_eeg_data(data, n_augmented=5):
    """
    Augment EEG data using time-frequency domain transformations
    """
    augmented_data = []
    
    for i in range(len(data)):
        sample = data[i]
        # Original sample
        augmented_data.append(sample)
        
        # Time shifting
        for j in range(n_augmented):
            shift = np.random.randint(-20, 20)
            shifted = np.roll(sample, shift, axis=-1)
            augmented_data.append(shifted)
        
        # Amplitude scaling
        for j in range(n_augmented):
            scale = np.random.uniform(0.8, 1.2)
            scaled = sample * scale
            augmented_data.append(scaled)
        
        # Add Gaussian noise
        for j in range(n_augmented):
            noise = np.random.normal(0, 0.01, sample.shape)
            noisy = sample + noise
            augmented_data.append(noisy)
    
    return np.array(augmented_data)

def preprocess_eeg(raw_data):
    """
    Preprocess EEG data with improved filtering and artifact removal
    """
    log("Starting preprocessing of EEG data...")
    
    # Apply more sophisticated filtering
    log("Applying filters...")
    raw_data.filter(l_freq=1.0, h_freq=70.0)  # Band-pass filter
    raw_data.notch_filter(np.arange(50, 71, 50))  # Remove power line noise
    log("Filters applied.")
    
    # Apply ICA for artifact removal with more components
    log("Applying ICA for artifact removal...")
    
    # Determine optimal number of components
    n_components = min(15, len(raw_data.ch_names) - 1)  # Reduced to avoid instability
    
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        random_state=42,
        method='fastica',
        max_iter='auto'
    )
    
    # Fit ICA
    ica.fit(raw_data)
    
    # Find artifacts using signal properties
    sources = ica.get_sources(raw_data)
    source_data = sources.get_data()
    
    # Calculate metrics for each component
    scores = []
    for comp_idx in range(source_data.shape[0]):
        comp_data = source_data[comp_idx]
        # Calculate kurtosis and variance
        kurt = scipy.stats.kurtosis(comp_data)
        var = np.var(comp_data)
        # Combined score
        score = abs(kurt) * var
        scores.append(score)
    
    # Find components with extreme scores
    threshold = np.percentile(scores, 75)  # Remove top 25% most artifact-like components
    bad_idx = np.where(np.array(scores) > threshold)[0]
    
    if len(bad_idx) > 0:
        ica.exclude = bad_idx.tolist()
        log(f"Excluding {len(bad_idx)} ICA components")
    
    # Apply ICA
    ica.apply(raw_data)
    log("ICA applied and artifacts removed.")
    
    # Get data and standardize
    log("Standardizing the data...")
    data = raw_data.get_data()
    
    # Robust standardization
    scaler = RobustScaler()
    data_reshaped = data.reshape(data.shape[0], -1)
    data_scaled = scaler.fit_transform(data_reshaped.T).T
    data = data_scaled.reshape(data.shape)
    log("Data standardized.")
    
    # Create windows with better overlap
    window_size = 256
    overlap = window_size * 3 // 4  # 75% overlap for better temporal coverage
    
    windows = []
    for i in range(0, data.shape[1] - window_size, window_size - overlap):
        window = data[:, i:i + window_size]
        if window.shape[1] == window_size:
            # Add frequency domain features
            freqs = np.fft.rfftfreq(window_size)
            fft_vals = np.fft.rfft(window)
            window = np.concatenate([window, np.abs(fft_vals)], axis=-1)
            windows.append(window)
    
    windows = np.array(windows)
    
    # Match fMRI samples while preserving temporal patterns
    n_fmri_samples = 100  # Increased sample size
    if len(windows) > n_fmri_samples:
        indices = np.linspace(0, len(windows)-1, n_fmri_samples, dtype=int)
        windows = windows[indices]
    
    log(f"Data processed, shape: {windows.shape}")
    return windows

def load_single_class_dataset(base_url, fmri_directory):
    """Load dataset with single class training and test split"""
    log("Loading single-class EEG and fMRI datasets...")
    
    # Load only non-seizure data (class 0) - reduced to 2 files
    non_seizure_files = [f"chb01_{str(i).zfill(2)}.edf" for i in range(3, 5)]  # Using only 2 files (03 and 04)
    processed_data = []
    
    for file in non_seizure_files:
        try:
            log(f"Processing EEG file: {file}")
            data = preprocess_eeg(stream_eeg_data(f"{base_url}/{file}")[0])
            if data is not None:
                processed_data.append(data)
                log(f"Successfully processed {file}, shape: {data.shape}")
        except Exception as e:
            log(f"Error loading {file}: {str(e)}")
            continue
    
    if not processed_data:
        raise ValueError("No data could be loaded")
    
    # Stack all processed data
    all_eeg_data = np.concatenate(processed_data, axis=0)
    log(f"Total EEG data shape after processing: {all_eeg_data.shape}")
    
    # Load fMRI data
    fmri_data = load_fmri_data(fmri_directory)
    if fmri_data is None:
        raise ValueError("Could not load fMRI data")
    
    # Ensure we have matching number of samples for training
    n_samples = min(len(all_eeg_data), len(fmri_data))
    log(f"Using {n_samples} samples from each modality")
    
    # Trim both datasets to the same size
    all_eeg_data = all_eeg_data[:n_samples]
    fmri_data = fmri_data[:n_samples]
    
    # Add channel dimension to EEG if needed
    if all_eeg_data.ndim == 3:
        all_eeg_data = np.expand_dims(all_eeg_data, axis=-1)
    
    # Create labels (all non-seizure for training)
    labels = np.zeros((n_samples, 2))
    labels[:, 0] = 1  # Set all to non-seizure [1, 0]
    
    # Use most data for training, small validation
    val_size = 1  # Just one sample for validation
    test_size = 1  # Just one sample for test
    train_size = n_samples - val_size - test_size
    
    # Training data (all non-seizure)
    X_train_eeg = all_eeg_data[:train_size]
    X_train_fmri = fmri_data[:train_size]
    y_train = labels[:train_size]
    
    # Validation data (non-seizure)
    X_val_eeg = all_eeg_data[train_size:train_size+val_size]
    X_val_fmri = fmri_data[train_size:train_size+val_size]
    y_val = labels[train_size:train_size+val_size]
    
    # Test data: one non-seizure, one seizure (reuse a sample as seizure)
    X_test_eeg = np.concatenate([all_eeg_data[-1:], all_eeg_data[:1]])  # Last and first samples
    X_test_fmri = np.concatenate([fmri_data[-1:], fmri_data[:1]])
    
    # Test labels: [non-seizure, seizure]
    y_test = np.array([[1, 0], [0, 1]])
    
    log(f"Final dataset shapes:")
    log(f"Train EEG: {X_train_eeg.shape}, Train fMRI: {X_train_fmri.shape}")
    log(f"Val EEG: {X_val_eeg.shape}, Val fMRI: {X_val_fmri.shape}")
    log(f"Test EEG: {X_test_eeg.shape}, Test fMRI: {X_test_fmri.shape}")
    
    return X_train_eeg, X_val_eeg, X_test_eeg, X_train_fmri, X_val_fmri, X_test_fmri, y_train, y_val, y_test

def preprocess_data(X_eeg, X_fmri):
    """Preprocess EEG and fMRI data to ensure correct shapes"""
    # Ensure EEG data is in the right shape (batch_size, height, width, channels)
    if len(X_eeg.shape) == 3:
        X_eeg = np.expand_dims(X_eeg, axis=-1)
    
    # Ensure fMRI data is in the right shape (batch_size, 1, height, width)
    if len(X_fmri.shape) == 3:
        X_fmri = np.expand_dims(X_fmri, axis=1)
    elif len(X_fmri.shape) == 4 and X_fmri.shape[3] == 1:
        X_fmri = np.transpose(X_fmri, (0, 3, 1, 2))  # Move channel dim to position 1
    
    # Print shapes for debugging
    log(f"EEG shape after preprocessing: {X_eeg.shape}")
    log(f"fMRI shape after preprocessing: {X_fmri.shape}")
    
    return X_eeg, X_fmri

def augment_data(X_eeg, X_fmri, y):
    """Augment training data with noise and transformations"""
    batch_size = X_eeg.shape[0]
    augmented_samples = 5  # Create 5 augmented versions of each sample
    
    # Initialize arrays for augmented data
    X_eeg_aug = np.zeros((batch_size * augmented_samples, *X_eeg.shape[1:]))
    X_fmri_aug = np.zeros((batch_size * augmented_samples, *X_fmri.shape[1:]))
    y_aug = np.zeros((batch_size * augmented_samples, *y.shape[1:]))
    
    for i in range(batch_size):
        # Original sample
        X_eeg_aug[i*augmented_samples] = X_eeg[i]
        X_fmri_aug[i*augmented_samples] = X_fmri[i]
        y_aug[i*augmented_samples] = y[i]
        
        # Generate augmented samples
        for j in range(1, augmented_samples):
            # EEG augmentation
            noise_level = np.random.uniform(0.05, 0.15)
            time_shift = np.random.randint(-20, 20)
            scale_factor = np.random.uniform(0.8, 1.2)
            
            # Apply augmentation to EEG
            aug_eeg = X_eeg[i] * scale_factor
            aug_eeg = np.roll(aug_eeg, time_shift, axis=1)
            aug_eeg += np.random.normal(0, noise_level, aug_eeg.shape)
            
            # fMRI augmentation
            noise_level = np.random.uniform(0.05, 0.15)
            rotation_angle = np.random.randint(-10, 10)
            
            # Apply augmentation to fMRI
            aug_fmri = X_fmri[i]
            aug_fmri = scipy.ndimage.rotate(aug_fmri[0], rotation_angle, axes=(0,1), reshape=False)
            aug_fmri = np.expand_dims(aug_fmri, 0)
            aug_fmri += np.random.normal(0, noise_level, aug_fmri.shape)
            
            # Store augmented samples
            X_eeg_aug[i*augmented_samples + j] = aug_eeg
            X_fmri_aug[i*augmented_samples + j] = aug_fmri
            y_aug[i*augmented_samples + j] = y[i]
    
    return X_eeg_aug, X_fmri_aug, y_aug

def load_and_preprocess_large_dataset(eeg_files, fmri_files, chunk_size=1000):
    """Load and preprocess large datasets in chunks to manage memory"""
    X_eeg_chunks = []
    X_fmri_chunks = []
    y_chunks = []
    
    for i in range(0, len(eeg_files), chunk_size):
        # Load and preprocess EEG data chunk
        eeg_chunk = []
        for file in eeg_files[i:i+chunk_size]:
            data = preprocess_eeg(mne.io.read_raw_edf(file, preload=True))
            eeg_chunk.append(data)
        X_eeg_chunks.append(np.array(eeg_chunk))
        
        # Load and preprocess fMRI data chunk
        fmri_chunk = []
        for file in fmri_files[i:i+chunk_size]:
            data = load_fmri_data(file)
            fmri_chunk.append(data)
        X_fmri_chunks.append(np.array(fmri_chunk))
        
        # Clear memory
        gc.collect()
        K.clear_session()
    
    return np.concatenate(X_eeg_chunks), np.concatenate(X_fmri_chunks)

def create_model(eeg_shape, fmri_shape):
    """Create a simpler model architecture better suited for small datasets"""
    
    # EEG branch with reduced complexity
    eeg_input = Input(shape=eeg_shape)
    x1 = GaussianNoise(0.05)(eeg_input)
    
    # Simplified EEG processing
    x1 = Conv2D(16, (3, 3), padding='same', kernel_regularizer=l2(1e-4))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling2D((2, 2))(x1)
    x1 = Dropout(0.3)(x1)
    
    x1 = Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(1e-4))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling2D((2, 2))(x1)
    x1 = Dropout(0.3)(x1)
    
    x1 = Flatten()(x1)
    x1 = Dense(64, kernel_regularizer=l2(1e-4))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Dropout(0.3)(x1)
    
    # fMRI branch with reduced complexity
    fmri_input = Input(shape=fmri_shape)
    x2 = Reshape((246, 246, 1))(fmri_input)
    x2 = GaussianNoise(0.05)(x2)
    
    x2 = Conv2D(16, (3, 3), padding='same', kernel_regularizer=l2(1e-4))(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = MaxPooling2D((2, 2))(x2)
    x2 = Dropout(0.3)(x2)
    
    x2 = Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(1e-4))(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = MaxPooling2D((2, 2))(x2)
    x2 = Dropout(0.3)(x2)
    
    x2 = Flatten()(x2)
    x2 = Dense(64, kernel_regularizer=l2(1e-4))(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = Dropout(0.3)(x2)
    
    # Merge with reduced complexity
    merged = concatenate([x1, x2])
    x = Dense(32, activation='relu', kernel_regularizer=l2(1e-4))(merged)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Output layer
    output = Dense(2, activation='softmax')(x)
    
    model = Model(inputs=[eeg_input, fmri_input], outputs=output)
    
    # Compilation with modified learning rate
    initial_learning_rate = 5e-4  # Reduced learning rate
    decay_steps = 500
    decay_rate = 0.95
    learning_rate_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps,
        decay_rate
    )
    optimizer = Adam(learning_rate=learning_rate_schedule, clipnorm=1.0)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=[
            'categorical_accuracy',
            f1_m,
            precision_m, 
            recall_m,
            g_mean
        ]
    )
    
    return model

def train_model(model, X_train, y_train, X_val, y_val, X_test, y_test):
    """Train the model with improved memory efficiency and HDF5 saving"""
    # Create balanced data generators
    train_generator = BalancedBatchGenerator(X_train[0], X_train[1], y_train, batch_size=32)
    val_generator = BalancedBatchGenerator(X_val[0], X_val[1], y_val, batch_size=32)
    
    # Create model directory if it doesn't exist
    os.makedirs('model_checkpoints', exist_ok=True)
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
        ModelCheckpoint(
            'model_checkpoints/Tri-SeizureDualNet.hdf5',
            monitor='val_loss',
            save_best_only=True,
            save_format='h5'
        )
    ]
    
    # Train model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=100,
        callbacks=callbacks,
        workers=4,
        use_multiprocessing=True
    )
    
    # Save final model in HDF5 format with the specified name
    model.save('Tri-SeizureDualNet.hdf5', save_format='h5')
    
    return history

# Updated metrics with percentage scaling
def precision_m(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(K.round(y_pred), 'float32')
    true_positives = K.sum(y_true * y_pred)
    predicted_positives = K.sum(y_pred)
    return (true_positives / (predicted_positives + K.epsilon())) * 100

def recall_m(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(K.round(y_pred), 'float32')
    true_positives = K.sum(y_true * y_pred)
    possible_positives = K.sum(y_true)
    return (true_positives / (possible_positives + K.epsilon())) * 100

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return (2 * (precision * recall) / (precision + recall + K.epsilon()))

def g_mean(y_true, y_pred):
    y_pred = K.cast(K.round(y_pred), 'int32')
    y_true = K.cast(y_true, 'int32')
    cm = tf.math.confusion_matrix(K.flatten(y_true), K.flatten(y_pred), num_classes=2)
    cm = tf.cast(cm, 'float32')
    
    # Add smoothing to prevent zero division
    smoothing = 1.0
    sensitivity = (cm[1,1] + smoothing) / (cm[1,1] + cm[1,0] + smoothing)
    specificity = (cm[0,0] + smoothing) / (cm[0,0] + cm[0,1] + smoothing)
    return K.sqrt(sensitivity * specificity) * 100

# Utility function to log messages
def log(message):
    print(f"[LOG]: {message}")

# Main pipeline
if __name__ == "__main__":
    try:
        # Enable mixed precision for better performance
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
        # Load single-class dataset
        base_url = "https://physionet.org/files/chbmit/1.0.0/chb01/"
        fmri_directory = "fMRI Data"
        
        log("Loading single-class EEG and fMRI datasets...")
        X_train_eeg, X_val_eeg, X_test_eeg, X_train_fmri, X_val_fmri, X_test_fmri, y_train, y_val, y_test = load_single_class_dataset(base_url, fmri_directory)
        
        # Preprocess data
        X_train_eeg, X_train_fmri = preprocess_data(X_train_eeg, X_train_fmri)
        X_val_eeg, X_val_fmri = preprocess_data(X_val_eeg, X_val_fmri)
        X_test_eeg, X_test_fmri = preprocess_data(X_test_eeg, X_test_fmri)
        
        # Create and compile model with correct shapes
        model = create_model(
            eeg_shape=(23, 385, 1),  # Original EEG shape
            fmri_shape=(1, 246, 246)  # Original fMRI shape
        )
        
        # Train model
        train_model(model, [X_train_eeg, X_train_fmri], y_train, [X_val_eeg, X_val_fmri], y_val, [X_test_eeg, X_test_fmri], y_test)
        
    except Exception as e:
        log(f"Error during execution: {str(e)}")
        log(f"Traceback: {traceback.format_exc()}")
    finally:
        tf.keras.backend.clear_session()
        gc.collect()
