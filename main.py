import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
torch.backends.cudnn.enabled = False

# 设置随机种子
seed = 1444
np.random.seed(seed)
torch.manual_seed(seed)


# 定义LSTM模型
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# 定义自定义数据集
class BikeDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __getitem__(self, index):
        x = self.data[index:index + self.seq_length, :]
        y = self.data[index + self.seq_length, :]
        return x, y

    def __len__(self):
        return len(self.data) - self.seq_length


# 设置训练参数
input_size = 1
hidden_size = 128
num_layers = 2
output_size = 1
num_epochs = 200
batch_size = 32
learning_rate = 0.001
seq_length = 24  # 输入序列长度，即预测当前小时的自行车数量，基于前面24小时的数据

# 加载数据
data = np.load('bike_data.npy')  # 加载纽约公共自行车站点小时聚合数据
train_size = int(0.8 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]

# 将数据转换为PyTorch张量
train_data_tensor = torch.Tensor(train_data).view(-1, 1)
test_data_tensor = torch.Tensor(test_data).view(-1, 1)

# 数据归一化
train_mean = train_data_tensor.mean()
train_std = train_data_tensor.std()
train_data_tensor = (train_data_tensor - train_mean) / train_std
test_data_tensor = (test_data_tensor - train_mean) / train_std

# 创建数据加载器
train_dataset = BikeDataset(train_data_tensor, seq_length)
test_dataset = BikeDataset(test_data_tensor, seq_length)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 检查GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建模型实例
model = LSTM(input_size, hidden_size, num_layers, output_size).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
train_losses = []
test_losses=[]
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        # 前向传播
        outputs = model(inputs)
        # 计算损失
        loss = criterion(outputs, labels)
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if(i + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item():.4f}')

    train_losses.append(loss.item())


    # 测试模型

    predicted_values = []
    actual_values = []
    model.eval()
    with torch.no_grad():
        test_loss = 0
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            test_loss_temp=criterion(outputs, labels).item()
            test_loss +=test_loss_temp
            predicted_values.extend(outputs.squeeze().tolist())
            actual_values.extend(labels.squeeze().tolist())
        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        print(f'Average Test Loss: {avg_test_loss:.4f}')

torch.save(model.state_dict(), 'bike_lstm_model.pth')
# 绘制损失折线图
plt.plot(train_losses,label='train_losses')
plt.plot(test_losses,label='test_losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.show()
plt.savefig('./Training_Loss.png')
# 绘制预测结果和真实结果折线图
plt.plot(predicted_values, label='Predicted')
plt.plot(actual_values, label='Actual')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('435 Station ID--Predicted vs Actual')
plt.legend()
plt.show()
plt.savefig('./Compare.png')