import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import copy
import nfode

dt = 0.01
data = np.load('data.npz')
t_data = data['t_data']
x_data = data['x_data_noised']
model = torch.load('model.pt')
x_preds = nfode.predict(model, t_data, x_data.T, dt)

params, flat_params, param_types, param_shapes = nfode.flatten_params(model)

print(param_types[67])