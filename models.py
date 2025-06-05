#models.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """
    基础CNN模型 (4层结构，约6万参数)
    结构符合论文描述：
    - 2个卷积层 (5x5 kernel)
    - 2个全连接层
    """

    def __init__(self):
        super(Net, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)  # 输入1通道，输出32通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        # 全连接层
        self.fc1 = nn.Linear(64 * 4 * 4, 512)  # 输入尺寸计算：两次池化后28x28 → 7x7 → 4x4
        self.fc2 = nn.Linear(512, 10)
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 输出尺寸: (32, 12, 12)
        x = self.pool(F.relu(self.conv2(x)))  # 输出尺寸: (64, 4, 4)
        x = x.view(-1, 64 * 4 * 4)  # 展平
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_param_count(self):
        """返回模型参数总数"""
        return sum(p.numel() for p in self.parameters())


class AlexNet(nn.Module):
    """
    改进的AlexNet模型 (8层结构，约6000万参数)
    根据论文描述调整：
    - 4个卷积层 (3x3 kernel)
    - 4个全连接层
    """

    def __init__(self):
        super(AlexNet, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        # 全连接层
        self.fc1 = nn.Linear(512 * 1 * 1, 4096)  # 修正输入维度
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1024)
        self.fc4 = nn.Linear(1024, 10)
        # 池化与Dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 64x14x14
        x = self.pool(F.relu(self.conv2(x)))  # 128x7x7
        x = self.pool(F.relu(self.conv3(x)))  # 256x3x3
        x = self.pool(F.relu(self.conv4(x)))  # 512x1x1
        x = x.view(-1, 512 * 1 * 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def get_param_count(self):
        return sum(p.numel() for p in self.parameters())

    def _reset_fc(self, flatten_size):
        """动态重置全连接层"""
        self.fc1 = nn.Linear(flatten_size, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1024)
        self.fc4 = nn.Linear(1024, 10)


class SplitModel(nn.Module):
    """
    支持分割训练的模型封装
    根据论文描述：在第二个卷积层后分割
    """

    def __init__(self, base_model, split_layer=2):
        """
        Args:
            base_model (nn.Module): 基础模型 (Net或AlexNet)
            split_layer (int): 分割层编号 (从1开始计数)
        """
        super(SplitModel, self).__init__()
        self.split_layer = split_layer
        self.ue_layers = nn.Sequential()
        self.server_layers = nn.Sequential()

        # 自动分割模型
        if isinstance(base_model, Net):
            self._split_net(base_model)
        elif isinstance(base_model, AlexNet):
            self._split_alexnet(base_model)
        else:
            raise ValueError("Unsupported base model type")

    def _split_net(self, model):
        """分割Net模型"""
        # UE侧: conv1 + pool + conv2 + pool
        self.ue_layers.add_module("conv1", model.conv1)
        self.ue_layers.add_module("pool1", model.pool)
        self.ue_layers.add_module("conv2", model.conv2)
        self.ue_layers.add_module("pool2", model.pool)
        # Server侧: fc1 + fc2
        self.server_layers.add_module("flatten", nn.Flatten())
        self.server_layers.add_module("fc1", model.fc1)
        self.server_layers.add_module("relu", nn.ReLU())
        self.server_layers.add_module("fc2", model.fc2)

    def _split_alexnet(self, model):
        """分割AlexNet模型"""
        # UE侧: conv1~conv2
        self.ue_layers.add_module("conv1", model.conv1)
        self.ue_layers.add_module("pool1", model.pool)
        self.ue_layers.add_module("conv2", model.conv2)
        self.ue_layers.add_module("pool2", model.pool)

        # 创建临时卷积层序列计算展平尺寸
        temp_conv = nn.Sequential(
            model.conv3,
            model.pool,
            model.conv4,
            model.pool
        )

        # 计算服务器端卷积后的展平尺寸
        test_input = torch.randn(1, 1, 28, 28)
        with torch.no_grad():
            ue_output = self.ue_layers(test_input)
            server_conv_output = temp_conv(ue_output)
            flatten_size = server_conv_output.view(1, -1).shape[1]

        # 重设全连接层输入维度
        model._reset_fc(flatten_size)

        # 构建服务器端完整层结构
        self.server_layers.add_module("conv3", model.conv3)
        self.server_layers.add_module("pool3", model.pool)
        self.server_layers.add_module("conv4", model.conv4)
        self.server_layers.add_module("pool4", model.pool)
        self.server_layers.add_module("flatten", nn.Flatten())
        self.server_layers.add_module("fc1", model.fc1)
        self.server_layers.add_module("relu1", nn.ReLU())
        self.server_layers.add_module("dropout1", model.dropout)
        self.server_layers.add_module("fc2", model.fc2)
        self.server_layers.add_module("relu2", nn.ReLU())
        self.server_layers.add_module("dropout2", model.dropout)
        self.server_layers.add_module("fc3", model.fc3)
        self.server_layers.add_module("relu3", nn.ReLU())
        self.server_layers.add_module("fc4", model.fc4)

    def forward_ue(self, x):
        """UE侧前向传播"""
        return self.ue_layers(x)

    def forward_server(self, x):
        """Server侧前向传播"""
        return self.server_layers(x)

    def get_split_output_shape(self, input_shape=(1, 28, 28)):
        """获取分割层输出形状（用于计算通信数据量）"""
        # 获取模型所在的设备
        device = next(self.parameters()).device
        test_input = torch.randn(1, *input_shape).to(device)
        with torch.no_grad():
            output = self.forward_ue(test_input)
        return output.shape[1:]  # 忽略batch维度

    def _calc_flatten_size(self):
        """动态计算全连接层输入维度"""
        test_input = torch.randn(1, 1, 28, 28)
        with torch.no_grad():
            output = self.ue_layers(test_input)
            output = self.server_layers[:6](output)  # 到最后一个池化层
            return output.view(-1).shape[0]



if __name__ == "__main__":
    # 验证模型结构与参数数量
    print("=== Net Model ===")
    net = Net()
    print(net)
    print(f"Total parameters: {net.get_param_count() / 1e3:.1f} K")

    print("\n=== AlexNet Model ===")
    alexnet = AlexNet()
    print(alexnet)
    print(f"Total parameters: {alexnet.get_param_count() / 1e6:.1f} M")

    print("\n=== Split Net ===")
    split_net = SplitModel(Net())
    test_input = torch.randn(1, 1, 28, 28)
    ue_output = split_net.forward_ue(test_input)
    print(f"UE侧输出形状: {ue_output.shape}")
    server_output = split_net.forward_server(ue_output)
    print(f"Server侧输出形状: {server_output.shape}")

    print("\n=== Split AlexNet ===")
    split_alex = SplitModel(AlexNet())
    ue_output = split_alex.forward_ue(test_input)
    print(f"UE侧输出形状: {ue_output.shape}")
    server_output = split_alex.forward_server(ue_output)
    print(f"Server侧输出形状: {server_output.shape}")