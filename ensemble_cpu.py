import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import nfode
import torch.jit
import random
from datetime import datetime

data = np.load('/home/liyilin/nfode/uq/data_set.npz')
t_data = data['t_data']
x_data = data['x_data']

device = torch.device('cpu')
x_data = torch.tensor(x_data, dtype=torch.float32).requires_grad_(True).to(device)
t_data = torch.tensor(t_data, dtype=torch.float32).requires_grad_(True).to(device)

test_data = np.load('/home/liyilin/nfode/uq/data_test.npz')
t_data_test = test_data['t_data']
x_data_test = test_data['x_data']

input_dim=2
output_dim=2
inner_dim=64
class ODEModel(nn.Module):
    def __init__(self):
        super(ODEModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, inner_dim)
        self.fc2 = nn.Linear(inner_dim, inner_dim)
        self.fc3 = nn.Linear(inner_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

k = 1
n = 1

error_matrix = []
params_matrix = []
epochs_vector = []
time_spent_vector = []
while n <= 300:
    model = ODEModel()
    nfode.initialize_params_uniform(model)
    model, epoch, time_spent = nfode.train(model, device, t_data, x_data, threshold=0.006, k=n, lr=0.01, milestones=[120,200,250,300,350,400], gamma=0.1)

    if epoch < 600:
        #采集参数
        params, flat_params, param_types, param_shapes = nfode.flatten_params(model)
        params_matrix.append(flat_params)

        #采集测试集误差
        x0 = x_data_test[:,0]
        x_preds_test = nfode.predict(model, t_data_test, x0)
        x_error = x_data_test - x_preds_test.T
        error_matrix.append(x_error)

        #采集训练用时
        epochs_vector.append(epoch)
        time_spent_vector.append(time_spent)

        n += 1

params_matrix = np.stack(params_matrix, axis = 0)
error_matrix = np.stack(error_matrix, axis = 2)
epochs_vector = np.array(epochs_vector)
time_spent_vector = np.array(time_spent_vector)

np.save('/home/liyilin/nfode/uq/params_matrix_cpu.npy', params_matrix)
np.save('/home/liyilin/nfode/uq/error_matrix_cpu.npy', error_matrix)
np.save('/home/liyilin/nfode/uq/epochs_vector_cpu.npy', epochs_vector)
np.save('/home/liyilin/nfode/uq/time_spent_vector_cpu.npy', time_spent_vector)