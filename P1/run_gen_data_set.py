import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from solve_m_height_cpp import solveMHeight_glpk
import time

from data_loader_saver import save_data


def gen_training_samples(n, k, m, num_samples, patience = 2048):
    total_iter = 0

    n_k_m_G_samples = []
    m_h_samples = []

    while len(n_k_m_G_samples) < num_samples:
        #create np.array of shape (k, n-k) with random integers between -100 and 100
        M = np.random.randint(-100, 100, size=(k, n-k))

        #create identity matrix of size k by k
        I_k = np.eye(k)

        #construct G matrix as I_k | M
        G = np.hstack((I_k, M))

        #solve for m_height
        m_h = solveMHeight_glpk(G, m)[0]

        if not np.isinf(m_h) and m_h is not None:
            n_k_m_G_samples.append((n, k, m, M))
            m_h_samples.append(m_h)
        
        total_iter += 1

        if total_iter - len(n_k_m_G_samples) > patience:
            print(f'Breaking after {total_iter} iterations with {len(n_k_m_G_samples)} samples collected.')
            break

    return n_k_m_G_samples, m_h_samples


def gen_training_data_set():
    n = 9
    k_list = [4,5,6]
    num_samples = 10000

    data_n_k_m_G_datapath = f"../data/project/training_samples_n_k_m_G.pkl"
    data_m_h_datapath = f"../data/project/training_samples_m_h.pkl"

    total_samples_n_k_m_G = []
    total_samples_m_h = []
    
    for k in k_list:
        for m in range(2, n-k+1):
            print(f'===> Generating data for n={n}, k={k}, m={m}')
            start_time = time.time()
            n_k_m_G_samples, m_h_samples = gen_training_samples(n, k, m, num_samples)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Execution time: {execution_time} seconds")

            total_samples_n_k_m_G.extend(n_k_m_G_samples)
            total_samples_m_h.extend(m_h_samples)

    save_data(data_n_k_m_G_datapath, data_m_h_datapath, total_samples_n_k_m_G, total_samples_m_h)


def gen_test_data_set():
    n = 9
    k_list = [4,5,6]
    num_samples = 128

    data_n_k_m_G_datapath = f"../data/project/test_samples_n_k_m_G.pkl"
    data_m_h_datapath = f"../data/project/test_samples_m_h.pkl"

    total_samples_n_k_m_G = []
    total_samples_m_h = []
    
    for k in k_list:
        for m in range(2, n-k+1):
            print(f'===> Generating data for n={n}, k={k}, m={m}')
            start_time = time.time()
            n_k_m_G_samples, m_h_samples = gen_training_samples(n, k, m, num_samples)
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"Execution time: {execution_time} seconds")

            total_samples_n_k_m_G.extend(n_k_m_G_samples)
            total_samples_m_h.extend(m_h_samples)

    save_data(data_n_k_m_G_datapath, data_m_h_datapath, total_samples_n_k_m_G, total_samples_m_h)


if __name__ == "__main__":
    # gen_test_data_set()
    gen_training_data_set()
    
    
            

            






