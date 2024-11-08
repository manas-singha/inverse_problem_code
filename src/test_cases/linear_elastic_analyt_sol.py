import numpy as np
import pandas as pd
#initiate the values
np.random.seed(42)
x = np.linspace(0.1,1,10)
E_values = np.random.uniform(0.1,20,1000)
q = 14
L = 1
A = 1
t = 0.5

u_values = []
term2 = np.sin((2*np.pi*x)/L)
for i in E_values:
    term1 = (q*L*L)/(4*i*A*np.pi*np.pi)
    term3 = ((t/i) - ((q*L)/(2*np.pi*i*A)))*x
    u = (term1*term2) +term3
    u_values.append(u)

column_names = ['x1', 'x2', 'x3', 'x4', 'x5','x6','x7','x8','x9','x10']
data = pd.DataFrame(u_values, columns = column_names)

data['k'] = E_values
data.to_csv('linear_elastic_data.csv',index=False)