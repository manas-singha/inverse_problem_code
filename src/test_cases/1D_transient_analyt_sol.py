import numpy as np
import pandas as pd

#fix the values
np.random.seed(42)
K = 1.25#heat conductivity
rho = 1.6
c = 1.2
x = np.random.uniform(0,1,10)#length of bar
t = np.random.uniform(0,1,10)# time 
kappaa = np.random.uniform(5,20,1000)#thermal velocity values(parameter)


# Function to compute k_i
def compute_ki(kappa):
    numerator = -4 * np.sqrt(2) * K**2 * np.pi * np.sqrt(K) * np.exp(-kappa / (2 * K))
    denominator = kappa**2 + 4 * K**2 * (np.pi)**2 * K
    return numerator / denominator

# Compute u(x,t) for each kappa
u_values = []
for kappa in kappaa:
    sum_terms = np.sqrt(2) * np.sin(np.pi * x) * np.exp(-K * (np.pi)**2 * t / (rho * c)) * compute_ki(kappa)

    exponential_terms = np.exp((kappa/ (2 * K)) * x - (kappa**2 * t / (4 * K * rho * c)))
    
    denominator_terms = (1 - np.exp((kappa / K) * (x - 1))) / (1 - np.exp(-kappa / K))
    
    u = exponential_terms * sum_terms + denominator_terms
    u_values.append(u)
    

column_labels = [f'x{i+1}' for i in range(10)]
data = pd.DataFrame(u_values, columns=column_labels)
data['k'] = kappaa

data.to_csv("1D_transient_analyt_sol.csv", index=False)


