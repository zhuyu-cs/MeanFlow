import torch
import lmdb
import pickle
import numpy as np
import os

class LMDBLatentsDataset(torch.utils.data.Dataset):
    """    
    Args:
        lmdb_path (str): LMDB dataset path.
        flip_prob (float): flip or upflip.
    """
    def __init__(self, lmdb_path, flip_prob=0.5):
        self.lmdb_path = lmdb_path
        self.flip_prob = flip_prob
        self.env = None  # Will be lazily initialized per-worker
        
        # Read length using a temporary connection (closed immediately)
        tmp_env = lmdb.open(lmdb_path,
                           readonly=True,
                           lock=False,
                           readahead=False,
                           meminit=False)
        with tmp_env.begin() as txn:
            self.length = int(txn.get('num_samples'.encode()).decode())
        tmp_env.close()
    
    def _init_env(self):
        """Lazily initialize LMDB environment (called once per worker process)."""
        if self.env is None:
            self.env = lmdb.open(self.lmdb_path,
                                readonly=True,
                                lock=False,
                                readahead=False,
                                meminit=False)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        self._init_env()  # Ensure env is initialized in this worker
        
        with self.env.begin() as txn:
            data = txn.get(f'{index}'.encode())
            if data is None:
                raise IndexError(f'Index {index} is out of bounds')
            
            data = pickle.loads(data)
            moments = data['moments']
            moments_flip = data['moments_flip']
            label = data['label']
            
            use_flip = torch.rand(1).item() < self.flip_prob
            
            moments_to_use = moments_flip if use_flip else moments
            
            moments_tensor = torch.from_numpy(moments_to_use).float()
            
            return moments_tensor, label
    
    def __del__(self):
        if hasattr(self, 'env') and self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass

