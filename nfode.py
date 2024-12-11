import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from datetime import datetime

# 导函数与时间有关的Euler_Step
def euler_step(model, ti, x_last, dt):
    dx = model(ti, x_last)
    x_next = x_last + dx * dt
    return x_next

# 导函数与时间无关的Euler_Step
def euler_step2(model, dt, x_last, m=1):
    dx = model(x_last)
    x_next = x_last + dx * dt
    return x_next

def vdpo(x, m=1):
    dx1 = m*(x[0] - 1/3*x[0]**3 -x[1])
    dx2 = 1/m*x[0]
    return np.array([dx1, dx2])


# 定义神经网络模型
class ODEModel(nn.Module):
    def __init__(self):
        super(ODEModel, self).__init__()
        self.fc1 = nn.Linear(3, 64)  # 输入为 x1, x2 和 t
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)  # 输出为 dx1 和 dx2

    def forward(self, t, x):
        if t.dim() == 0:
            t = t.unsqueeze(0)
        elif t.dim() == 1:
            t = t.unsqueeze(1)
        elif t.size(1) > 2:
            t = t[:, :2]
        x = torch.cat([x, t], dim=0)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def initialize_params_uniform(model, low=-0.2, high=0.2):
    with torch.no_grad():
        for param in model.parameters():
            # 生成均匀分布的随机数，并设置为参数
            param.uniform_(low, high)

# 主训练过程
def train(model, t_data, x_data, device, j=1, num_epochs=1200, dt=0.01, lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[800], gamma=0.2, last_epoch=-1)
    loss_fn = nn.MSELoss()
    x_data = x_data.T.clone().detach().requires_grad_(True).to(device)
    t_data = t_data.clone().detach().requires_grad_(True).unsqueeze(1).to(device)
    x_true_grads = []
    for i in range(1, len(x_data)):
        if i == 1:
            x_true_grad = (x_data[i] - x_data[i - 1]) - (x_data[i] - x_data[i - 1])
            x_true_grads.append(x_true_grad)
        else:
            x_true_grad = (x_data[i] - x_data[i - 1]) / (dt)
            x_true_grads.append(x_true_grad)
    x_true_grads = torch.stack(x_true_grads).squeeze()
    loss_curve = []
    real_loss_curve = []
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # 初始状态
        x_pred = x_data[0].clone().detach().requires_grad_(True).unsqueeze(0)
        x_preds = [x_pred]
        x_grads = []
        for i in range(1, len(t_data)):
            t = t_data[i - 1].requires_grad_(True)
            dx = model(t, x_pred).requires_grad_(True)
            x_pred = euler_step(model, t, x_pred, dt)
            x_preds.append(x_pred)
            x_grads.append(dx)

        # 计算损失
        x_preds = torch.stack(x_preds).squeeze()
        x_grads = torch.stack(x_grads).squeeze()
        x_true_grads = x_true_grads[:len(x_grads), :]
        loss = loss_fn(x_grads, x_true_grads)
        loss_curve.append(loss.cpu().detach().numpy())
        real_loss = loss_fn(x_preds, x_data)
        real_loss_curve.append(real_loss.cpu().detach().numpy())
        # 反向传播
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 100 == 0:
            current_time = datetime.now()
            current_time = current_time.strftime("%H:%M:%S")
            model_device = next(model.parameters()).device
            print(j, epoch + 1, '/', num_epochs, current_time, model_device, 'real loss:', real_loss.detach().numpy(), 'derivate loss:', loss.detach().numpy())
    return model, loss_curve, x_preds, real_loss_curve

def predict(model, t_data, x_data, dt):
    model.eval()
    x_data = torch.tensor(x_data, dtype=torch.float32).requires_grad_(True)
    t_data = torch.tensor(t_data, dtype=torch.float32).unsqueeze(1).requires_grad_(True)
    with torch.no_grad():
        x_pred = x_data[0].clone().detach().unsqueeze(0)
        x_preds = [x_pred]

        # 进行训练集内的预测
        for i in range(1, len(t_data)):
            t = t_data[i].unsqueeze(0)
            x_pred = euler_step(model, t, x_pred, dt)
            x_preds.append(x_pred)

    x_preds = torch.stack(x_preds).squeeze().numpy()
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

