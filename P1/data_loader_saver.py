import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from solve_m_height_cpp import solveMHeight_glpk
import time

def load_data(n_k_m_datapath, m_h_datapath):
    with open(n_k_m_datapath, 'rb') as f:
        n_k_m_G_samples = pickle.load(f)
    with open(m_h_datapath, 'rb') as f:
        m_heights_samples = pickle.load(f)
    return n_k_m_G_samples, m_heights_samples

def save_data(n_k_m_datapath, m_h_datapath, n_k_m_G_samples, m_heights_samples):
    with open(n_k_m_datapath, 'wb') as f:
        pickle.dump(n_k_m_G_samples, f)
    with open(m_h_datapath, 'wb') as f:
        pickle.dump(m_heights_samples, f)

