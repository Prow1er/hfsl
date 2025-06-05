import torch
import numpy as np
from torch.utils.data import DataLoader
from models import Net, AlexNet
from data_utils import PartitionedDataset


class FLTrainer:
    def __init__(self, num_users, train_set, test_set, model_type='Net',
                 lr=0.01, batch_size=32, local_epochs=5):
        """
        初始化联邦学习训练器

        参数：
            num_users: 用户总数
            train_set: 训练数据集
            test_set: 测试数据集
            model_type: 模型类型 ['Net', 'AlexNet']
            lr: 学习率
            batch_size: 本地训练批量大小
            local_epochs: 本地训练轮数
        """
        self.num_users = num_users
        self.lr = lr
        self.batch_size = batch_size
        self.local_epochs = local_epochs

        # 初始化全局模型
        self.model = self._init_model(model_type)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.communication_cost = 0  # 单位：字节
        self.model_param_count = sum(p.numel() for p in self.model.parameters())

        self.user_datasets = self._prepare_user_datasets(train_set)
        self.test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

    def _init_model(self, model_type):
        if model_type == 'Net':
            return Net()
        elif model_type == 'AlexNet':
            return AlexNet()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _prepare_user_datasets(self, train_set):
        # 假设已经预先划分了IID/非IID数据
        return [PartitionedDataset(train_set, indices) for indices in train_set.partitions]

    def _client_train(self, model, user_idx):
        """单个客户端本地训练"""
        model.train()
        model.to(self.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
        criterion = torch.nn.CrossEntropyLoss()

        # 加载用户数据
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

    def aggregate(self, local_weights):
        """联邦平均聚合"""
        global_weights = self.model.state_dict()
        for key in global_weights:
            global_weights[key] = torch.mean(
                torch.stack([w[key].float() for w in local_weights]),
                dim=0
            )
        self.model.load_state_dict(global_weights)

    def test(self):
        """全局模型测试"""
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, labels in self.test_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total

    def train_round(self, selected_users):
        # num_selected = len(selected_users)
        # round_comm = num_selected * self.model_param_count * 4  # 4 bytes per param
        # self.communication_cost += round_comm
        """执行一轮训练"""
        local_weights = []
        for user in selected_users:
            # 分发全局模型
            local_model = self._init_model(type(self.model).__name__)
            local_model.load_state_dict(self.model.state_dict())

            # 本地训练
            local_weight = self._client_train(local_model, user)
            local_weights.append(local_weight)

        # 模型聚合
        self.aggregate(local_weights)
        return self.test()


if __name__ == "__main__":
    # 示例用法
    from data_utils import load_mnist, noniid_partition

    # 加载数据
    train_set, test_set = load_mnist()
    train_set.partitions = noniid_partition(train_set, num_users=100)

    # 初始化训练器
    fl = FLTrainer(
        num_users=100,
        train_set=train_set,
        test_set=test_set,
        model_type='Net'
    )

    # 模拟训练过程
    for round in range(10):
        selected = np.random.choice(100, size=10, replace=False)
        acc = fl.train_round(selected)
        print(f"Round {round + 1}, Test Accuracy: {acc:.4f}")