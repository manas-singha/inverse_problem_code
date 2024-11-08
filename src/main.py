import os
import pandas as pd
import sys
from data_preprocessing import data_preprocessing
from decoder_model import decoder
from vae_arch import vae_kl, vae_js,vae_wd
from plots_results_training import result_plots,plot
import pickle
import numpy as np


# Define output directories
plot_dir = r"C:\Users\MANAS SINGHA\OneDrive\Pictures\Documents\output\train_plots" # update paths as necessary
result_dir = r"C:\Users\MANAS SINGHA\OneDrive\Pictures\Documents\output\results"

# Create directories if they don't exist
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)

# Load the data
data = np.fromfile(r"C:\Users\MANAS SINGHA\OneDrive\Desktop\result.bin", dtype=np.float32)
data = pd.DataFrame(data, columns=["feature"])


# data = pd.read_csv(r"C:\Users\MANAS SINGHA\OneDrive\Pictures\Documents\codefile\venv1\ahazra-vae-b283497110df\src\data\1D_transient_analytical_solution.csv")
#data preprocessing , update the percentage as required
u_train_noisy, u_test_noisy, u_train, u_test, k_train, k_test, k_train_prior, k_test_prior = data_preprocessing(data, percentage=0.25)

#training the decoder model with non noisy data as a surrogate
history_decoder = decoder(u_train,u_test,k_train,k_test)
plot_path = os.path.join(plot_dir, 'decoder_training.png')
plot(history_decoder,save_path=plot_path)

def process_vae(vae_function, name):
    history_vae, z_mean, z_log_var, z_mean_t, z_log_var_t = vae_function(u_train_noisy, u_test_noisy, k_train_prior, k_test_prior)
    plot_path = os.path.join(plot_dir, f"{name}_plot.png")
    plot(history_vae, save_path=plot_path)
    result_plot_path_1 = os.path.join(result_dir, f"{name}_train_results.png")
    result_plot_path_2 = os.path.join(result_dir, f"{name}_test_results.png")
    mse, r2, mse_t, r2_t = result_plots(z_mean, z_log_var, z_mean_t, z_log_var_t, k_train, k_test, save_path_train=result_plot_path_1, save_path_test=result_plot_path_2)
    return mse, r2, mse_t, r2_t
def process_vae_2(vae_function, name):
    history_vae, z_mean, z_log_var, z_mean_t, z_log_var_t = vae_function(u_train_noisy, u_test_noisy,k_train,k_test, k_train_prior, k_test_prior)
    plot_path = os.path.join(plot_dir, f"{name}_plot.png")
    plot(history_vae, save_path=plot_path)
    result_plot_path_1 = os.path.join(result_dir, f"{name}_train_results.png")
    result_plot_path_2 = os.path.join(result_dir, f"{name}_test_results.png")
    mse, r2, mse_t, r2_t = result_plots(z_mean, z_log_var, z_mean_t, z_log_var_t, k_train, k_test, save_path_train=result_plot_path_1, save_path_test=result_plot_path_2)
    return mse, r2, mse_t, r2_t

# Process VAE models
mse_k, r2_k, mse_t_k, r2_t_k = process_vae(vae_kl, "vae_kl")
mse_j, r2_j, mse_t_j, r2_t_j = process_vae_2(vae_js, "vae_js")
mse_w, r2_w, mse_t_w, r2_t_w = process_vae_2(vae_wd, "vae_wd")

# Print results
print(f"VAE KL - MSE for train data is {mse_k} and R2 score is {r2_k}")
print(f"VAE KL - MSE for test data is {mse_t_k} and R2 score is {r2_t_k}")
print(f"VAE JS - MSE for train data is {mse_j} and R2 score is {r2_j}")
print(f"VAE JS - MSE for test data is {mse_t_j} and R2 score is {r2_t_j}")
print(f"VAE WD - MSE for train data is {mse_w} and R2 score is {r2_w}")
print(f"VAE WD - MSE for test data is {mse_t_w} and R2 score is {r2_t_w}")
