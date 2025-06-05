#hsfl.py
import torch
import numpy as np
from torch.utils.data import DataLoader
from models import SplitModel, Net, AlexNet
from data_utils import PartitionedDataset


class HSFLTrainer:
    def __init__(self, num_users, train_set, test_set, model_type='Net',
                 split_ratio=0.5, lr=0.01, batch_size=32, local_epochs=5):
        """
        初始化HSFL训练器

        参数：
            split_ratio: 分割训练用户比例
        """
        self.num_users = num_users
        self.split_ratio = split_ratio
        self.lr = lr
        self.batch_size = batch_size
        self.local_epochs = local_epochs

        # 初始化全局模型
        self.base_model = self._init_base_model(model_type)
        self.global_model = SplitModel(self.base_model)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model.to(self.device)

        # 准备用户数据集
        self.user_datasets = self._prepare_user_datasets(train_set)
        self.test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

        # 记录切割层输出形状（用于计算通信开销）
        self.split_layer_shape = self.global_model.get_split_output_shape()

    def _init_base_model(self, model_type):
        if model_type == 'Net':
            return Net()
        elif model_type == 'AlexNet':
            return AlexNet()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _prepare_user_datasets(self, train_set):
        return [PartitionedDataset(train_set, indices) for indices in train_set.partitions]

    def get_global_model_state(self):
        """返回当前全局模型的深拷贝状态"""
        return {k: v.detach().clone() for k, v in self.global_model.state_dict().items()}

    def _federated_train(self, model, user_idx):
        """联邦训练模式（完整模型训练）"""
        model.train()
        model.to(self.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
        criterion = torch.nn.CrossEntropyLoss()

        loader = DataLoader(
            self.user_datasets[user_idx],
            batch_size=self.batch_size,
            shuffle=True
        )

        for _ in range(self.local_epochs):
            for data, labels in loader:
                data, labels = data.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        return model.state_dict()

    def _split_train(self, ue_model, server_model, user_idx):
        """分割训练模式"""
        ue_model.train()
        server_model.train()
        ue_model.to(self.device)
        server_model.to(self.device)
        criterion = torch.nn.CrossEntropyLoss()

        # 用户端优化器（仅优化UE侧模型）
        ue_optim = torch.optim.SGD(ue_model.parameters(), lr=self.lr)

        loader = DataLoader(
            self.user_datasets[user_idx],
            batch_size=self.batch_size,
            shuffle=True
        )

        for _ in range(self.local_epochs):
            for data, labels in loader:
                data, labels = data.to(self.device), labels.to(self.device)

                # UE侧前向传播（启用梯度追踪）
                data.requires_grad = True
                ue_output = ue_model.forward_ue(data)
                ue_output.retain_grad()  # 确保可以获取到梯度

                # Server侧前向+反向传播
                server_output = server_model.forward_server(ue_output)
                loss = criterion(server_output, labels)
                loss.backward(retain_graph=True)

                # 获取 server 对 ue_output 的梯度
                server_grad = ue_output.grad.detach()

                # 反向传播到UE模型
                ue_output.backward(server_grad)
                ue_optim.step()
                ue_optim.zero_grad()

        return ue_model.ue_layers.state_dict()

    def aggregate(self, fed_weights, split_weights):
        """混合聚合策略"""
        global_ue = self.global_model.ue_layers.state_dict()
        global_server = self.global_model.server_layers.state_dict()

        # 聚合联邦训练更新
        if fed_weights:
            for key in global_ue:
                global_ue[key] = torch.mean(
                    torch.stack([w[key] for w in fed_weights]),
                    dim=0
                )

        # 聚合分割训练更新（仅UE侧）
        if split_weights:
            for key in global_ue:
                global_ue[key] = torch.mean(
                    torch.stack([w[key] for w in split_weights]),
                    dim=0
                )

        # 更新全局模型
        self.global_model.ue_layers.load_state_dict(global_ue)
        self.global_model.server_layers.load_state_dict(global_server)

    def test(self):
        """全局模型测试"""
        self.global_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in self.test_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.global_model.forward_server(
                    self.global_model.forward_ue(data)
                )
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total

    def train_round(self, selected_users, split_users):
        """执行一轮HSFL训练"""
        fed_weights = []
        split_weights = []

        for user in selected_users:
            if user in split_users:
                # 分割训练用户
                ue_model = SplitModel(self._init_base_model(type(self.base_model).__name__))
                ue_model.ue_layers.load_state_dict(self.global_model.ue_layers.state_dict())
                server_model = SplitModel(self._init_base_model(type(self.base_model).__name__))
                server_model.server_layers.load_state_dict(self.global_model.server_layers.state_dict())

                # 分割训练
                ue_weights = self._split_train(ue_model, server_model, user)
                split_weights.append(ue_weights)
            else:
                # 联邦训练用户
                full_model = self._init_base_model(type(self.base_model).__name__)
                # 将 SplitModel 的参数还原为基础模型格式
                state_dict = {}

                # 加载 UE 层参数
                for k, v in self.global_model.ue_layers.state_dict().items():
                    key = k.replace("ue_layers.", "") if k.startswith("ue_layers.") else k
                    state_dict[key] = v

                # 加载 Server 层参数
                for k, v in self.global_model.server_layers.state_dict().items():
                    key = k.replace("server_layers.", "") if k.startswith("server_layers.") else k
                    state_dict[key] = v

                full_model.load_state_dict(state_dict)
                full_weights = self._federated_train(full_model, user)
                fed_weights.append(full_weights)

        # 模型聚合
        self.aggregate(fed_weights, split_weights)
        return self.test()




if __name__ == "__main__":
    from data_utils import load_mnist, noniid_partition

    # 加载数据
    train_set, test_set = load_mnist()
    train_set.partitions = noniid_partition(train_set, num_users=100)

    # 初始化HSFL训练器
    hsfl = HSFLTrainer(
        num_users=100,
        train_set=train_set,
        test_set=test_set,
        model_type='AlexNet',
        split_ratio=0.5
    )

    # 模拟训练过程（随机选择用户）
    for round in range(10):
        selected = np.random.choice(100, size=10, replace=False)
        split_users = np.random.choice(selected, size=int(10 * 0.5), replace=False)
        acc = hsfl.train_round(selected, split_users)
        print(f"Round {round + 1}, Test Accuracy: {acc:.4f}")
        print(f"Split Layer Shape: {hsfl.split_layer_shape}")  # 显示切割层信息
