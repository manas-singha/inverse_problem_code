import pandas as pd
import numpy as np
np.random.seed(42)
k_values = np.random.rand(1000) * 200 + 10
x = np.linspace(0.0,0.9,10)
t_values = []
for k in k_values:
    temp = 1000.0*((1-x)/k+1.0/12.5)+298
    t_values.append(temp)

column_labels = [f'x{i+1}' for i in range(10)]
data = pd.DataFrame(t_values, columns=column_labels)
data['k'] = k_values
print(data)
data.to_csv(r"C:\Users\rajes\Desktop\UQ_VAE_WD_Project\Data_generated\1D_heat.csv",index=False)