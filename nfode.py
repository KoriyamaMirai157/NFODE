import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy

# 导函数与时间无关的Euler_Step
def euler_step(model, dt, x_last, m=1):
    dx = model(x_last)
    x_next = x_last + dx * dt
    return x_next

def initialize_params_uniform(model, low=-0.2, high=0.2):
    with torch.no_grad():
        for param in model.parameters():
            # 生成均匀分布的随机数，并设置为参数
            param.uniform_(low, high)

# train函数要求输入时已经为张量，其中x_data第0维为x的维度，第1维为时间点，第2维区分不同数据样本; t_data第0维区分不同数据样本，第1维为时间点
def train(model, device, t_data, x_data, num_epochs = 200, k=1, lr=0.01, milestones=[800], gamma=0.1):
    from datetime import datetime
    import random
    import copy
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma, last_epoch=-1)
    loss_fn = nn.MSELoss()
    x_true_grads = torch.zeros_like(x_data)
    for i in range(1, x_data.size(1)):
        x_true_grads[:, i, :] = (x_data[:, i, :] - x_data[:, i - 1, :]) / (t_data[:, i] - t_data[:, i - 1])
    x_true_grads.to(device)
    loss_curve = []
    real_loss_curve = []
    best_model = copy.deepcopy(model)
    best_real_loss = 100
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        n = random.randint(0, x_true_grads.size(2) - 1)
        t = t_data[n, :].to(device)
        x = x_data[:, :, n].to(device)
        x_preds = torch.zeros_like(x).to(device)
        x_preds[:, 0] = x[:, 0].clone().detach().requires_grad_(True)
        x_grads = torch.zeros_like(x).to(device)
        for i in range(1, len(t)):
            dt = t[i] - t[i - 1]
            x_pred = x_preds[:, i - 1].clone()
            dx = model(x_pred).requires_grad_(True)
            x_pred = euler_step(model, dt, x_pred)
            x_preds[:, i] = x_pred
            x_grads[:, i] = dx
        loss = loss_fn(x_grads, x_true_grads[:, :, n])
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()
        loss = loss.cpu().detach().numpy()
        loss_curve.append(loss)
        real_loss = loss_fn(x_preds, x)
        real_loss = real_loss.cpu().detach().numpy()
        real_loss_curve.append(real_loss)

        if real_loss <= best_real_loss:
            best_model = copy.deepcopy(model)
            best_real_loss = real_loss

        if (epoch + 1) % 10 == 0:
            current_time = datetime.now()
            current_time = current_time.strftime("%H:%M:%S")
            model_device = next(model.parameters()).device
            print(k, epoch + 1, '/', num_epochs, current_time, model_device, 'best real loss:', best_real_loss)
    
    best_model.to('cpu')
    return best_model, x_preds, loss_curve, real_loss_curve

#predict函数不要求输入时已经为张量, t_data为时间点(1*n维数组), x0为初值(n*1维数组)
def predict(model, t_data, x0):
    model.eval()
    x0 = torch.tensor(x0, dtype=torch.float32).requires_grad_(True)
    t_data = torch.tensor(t_data, dtype=torch.float32).unsqueeze(1).requires_grad_(True)
    with torch.no_grad():
        x_pred = x0.clone().detach().unsqueeze(0)
        x_preds = [x_pred]

        for i in range(1, len(t_data)):
            dt = t_data[i] - t_data[i - 1]
            x_pred = euler_step(model, dt, x_pred)
            x_preds.append(x_pred)

    x_preds = torch.stack(x_preds).squeeze().cpu().numpy()
    return x_preds

def flatten_params(model):
    # Step 1: 访问模型中的所有参数并展平
    params = {}
    flat_params = []
    param_types = []
    param_shapes = {}

    for idx, (name, param) in enumerate(model.named_parameters()):
        param_array = param.detach().numpy()
        params[name] = param_array
        param_shapes[name] = param.shape

        # 拉平参数并拼接到flat_params中
        flat_params.append(param_array.ravel())
        # 生成与展平后的参数数量一致的类型信息
        param_type = 'weight' if 'weight' in name else 'bias'
        param_types.append(np.full(param_array.size, param_type, dtype='<U16'))

    # 将所有列表元素按顺序拼接成向量
    flat_params = np.concatenate(flat_params)
    param_types = np.concatenate(param_types)

    return params, flat_params, param_types, param_shapes


def update_model_params(model, modified_params, param_shapes):
    modified_model = copy.deepcopy(model)
    start_idx = 0

    # 使用 model.named_parameters() 逐个参数替换
    for name, param in modified_model.named_parameters():
        shape = param_shapes[name]
        param_size = np.prod(shape)
        param_data = modified_params[start_idx:start_idx + param_size]

        # 创建新的参数张量并确保形状正确
        param_tensor = torch.tensor(param_data).reshape(shape)

        # 通过 data.copy_() 更新参数
        param.data.copy_(param_tensor)

        # 更新起始索引
        start_idx += param_size

    return modified_model

