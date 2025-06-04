import os
import numpy as np
import pandas as pd
import random

# Clinical feature encoding dictionary
CLINICAL_DICT = {
    'Gender': {'Male': 0, 'Female': 1, 'missing': 2},
    'Smoking': {'No': 0, 'Yes': 1, 'stopped': 2, 'missing': 3}
}

class NMILDataset:
    """Multiple Instance Learning Dataset for medical imaging with clinical data"""
    
    def __init__(self, dir_embeddings, wsi_list, clinical_df_path, 
                 clinical_params, only_clinical=False):
        """
        Initialize dataset
        Args:
            dir_embeddings: Path to embedding files
            wsi_list: List of WSI (Whole Slide Image) IDs
            clinical_df_path: Path to clinical data Excel file
            clinical_params: List of clinical parameters to use
            only_clinical: If True, use only clinical data (no embeddings)
        """
        self.dir_embeddings = dir_embeddings
        self.only_clinical = only_clinical
        self.clinical_params = clinical_params
        self.clinical_df = pd.read_excel(clinical_df_path)
        
        # Dictionary mapping patient ID to embedding indices
        self.patient_to_indices = {}
        self.embeddings = []
        self.patient_ids = []
        self.clinical_data = []
        
        self._load_data(wsi_list)
        self.indexes = np.arange(len(self.embeddings) if not only_clinical else len(self.patient_to_indices))
    
    def _load_data(self, wsi_list):
        """Load embeddings and clinical data"""
        idx = 0
        
        for patient_id in wsi_list:
            embedding_path = os.path.join(self.dir_embeddings, f"{patient_id}.npy")
            
            # Skip if embedding file doesn't exist
            if not os.path.exists(embedding_path):
                continue
            
            # Load embeddings for this patient
            embeddings = np.load(embedding_path)
            
            # Map patient to embedding indices
            if patient_id not in self.patient_to_indices:
                self.patient_to_indices[patient_id] = []
            
            # Add each embedding with its index
            for emb in embeddings:
                self.patient_to_indices[patient_id].append(idx)
                self.embeddings.append(emb)
                self.patient_ids.append(patient_id)
                idx += 1
            
            # Extract clinical features for this patient
            clinical_features = self._extract_clinical_features(patient_id)
            
            # Replicate clinical data for each embedding of this patient
            for _ in range(len(embeddings)):
                self.clinical_data.append(clinical_features)
    
    def _extract_clinical_features(self, patient_id):
        """Extract and encode clinical features for a patient"""
        features = []
        patient_row = self.clinical_df[self.clinical_df['SID'] == int(patient_id)]
        
        for param in self.clinical_params:
            value = patient_row[param].iloc[0]
            
            # Handle age as continuous variable
            if param == 'Yrs_age':
                features.append(float(value) if not pd.isna(value) else 0.0)
            else:
                # Handle categorical variables
                if pd.isna(value):
                    # Use 'missing' category
                    features.append(len(CLINICAL_DICT[param]) - 1)
                else:
                    features.append(CLINICAL_DICT[param][value])
        
        return features
    
    def __len__(self):
        return len(self.indexes)
    
    def __getitem__(self, index):
        """Get a single sample"""
        if self.only_clinical:
            # Return only clinical data
            return None, self.clinical_data[self.indexes[index]]
        else:
            # Return embedding and clinical data
            embedding = self.embeddings[self.indexes[index]]
            clinical = self.clinical_data[self.indexes[index]]
            return embedding, clinical

class NMILDataGenerator:
    """Data generator for batch processing of MIL bags"""
    
    def __init__(self, dataset, batch_size=1, shuffle=True, max_instances=100):
        """
        Initialize data generator
        Args:
            dataset: NMILDataset instance
            batch_size: Number of bags per batch
            shuffle: Whether to shuffle patient order
            max_instances: Maximum patches per bag (for memory management)
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_instances = max_instances
        
        # Get unique patient IDs
        self.patient_ids = list(self.dataset.patient_to_indices.keys())
        self.indexes = np.arange(len(self.patient_ids))
        
        self._idx = 0
        self._reset()
    
    def __len__(self):
        """Number of batches per epoch"""
        return len(self.indexes) // self.batch_size + bool(len(self.indexes) % self.batch_size)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        """Generate next batch"""
        if self._idx >= len(self.indexes):
            self._reset()
            raise StopIteration()
        
        # Get patient ID for current batch
        patient_id = self.patient_ids[self.indexes[self._idx]]
        embedding_indices = self.dataset.patient_to_indices[patient_id]
        
        # Get bag-level label (BCG failure prediction)
        label = self._get_bag_label(patient_id)
        
        # Sample instances if bag is too large
        if len(embedding_indices) > self.max_instances:
            embedding_indices = random.sample(embedding_indices, self.max_instances)
        elif len(embedding_indices) < 4:  # Ensure minimum bag size
            embedding_indices.extend(embedding_indices)
        
        # Collect embeddings and clinical data
        embeddings = []
        clinical_data = None
        
        for idx in embedding_indices:
            emb, clinical = self.dataset[idx]
            if not self.dataset.only_clinical:
                embeddings.append(emb)
            clinical_data = clinical  # Same for all instances in bag
        
        self._idx += self.batch_size
        
        # Return batch
        X = np.array(embeddings).astype('float32') if embeddings else None
        Y = np.array(label).astype('float32')
        clinical = np.array(clinical_data).astype('float32')
        
        return X, Y, clinical
    
    def _get_bag_label(self, patient_id):
        """Get one-hot encoded bag label"""
        patient_row = self.dataset.clinical_df[self.dataset.clinical_df['SID'] == int(patient_id)]
        bcg_failure = patient_row['BCG_failure'].iloc[0] == 'Yes'
        
        # One-hot encoding: [No failure, Failure]
        label = [0.0, 1.0] if bcg_failure else [1.0, 0.0]
        return label
    
    def _reset(self):
        """Reset generator for next epoch"""
        if self.shuffle:
            random.shuffle(self.indexes)
        self._idx = 0

