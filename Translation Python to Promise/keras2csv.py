import h5py as h5
import pandas as pd
name = "mnist_nn_64.h5"

f = h5.File(name)

for layer in f['model_weights']:
    pd.DataFrame(f['model_weights'][layer][layer]['kernel:0']).to_csv(f"mnist_{layer}_kernel_array.csv", index=False, header=False, sep='\n')
    pd.DataFrame(f['model_weights'][layer][layer]['bias:0']).to_csv(f"mnist_{layer}_bias_array.csv", index=False, header=False)
