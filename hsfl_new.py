#hsfl_new.py
import torch
import numpy as np
from torch.utils.data import DataLoader
from models import SplitModel, Net, AlexNet
from data_utils import PartitionedDataset


class HSFLTrainer:
    def __init__(self, num_users, train_set, test_set, model_type='Net',
                 split_ratio=0.5, lr=0.01, batch_size=32, local_epochs=5, num_challenges=10, proof_threshold=1e-4):
        """
        初始化HSFL训练器

        参数：
            split_ratio: 分割训练用户比例
            num_challenges: 验证使用的挑战样本数量
            proof_threshold: 验证阈值（MSE）
        """
        self.num_users = num_users
        self.split_ratio = split_ratio
        self.lr = lr
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.num_challenges = num_challenges
        self.proof_threshold = proof_threshold

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

        # 准备挑战数据
        self.challenge_data, self.challenge_labels = self._get_challenge_samples(test_set, num_challenges)

    def _init_base_model(self, model_type):
        if model_type == 'Net':
            return Net()
        elif model_type == 'AlexNet':
            return AlexNet()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _prepare_user_datasets(self, train_set):
        return [PartitionedDataset(train_set, indices) for indices in train_set.partitions]

    def _get_challenge_samples(self, test_set, num_challenges):
        """随机选择挑战样本"""
        challenge_loader = DataLoader(test_set, batch_size=num_challenges, shuffle=True)
        data, labels = next(iter(challenge_loader))
        return data.to(self.device), labels.to(self.device)

    def get_global_model_state(self):
        """返回当前全局模型的深拷贝状态"""
        return {k: v.detach().clone() for k, v in self.global_model.state_dict().items()}

    def _federated_train(self, model, user_idx):
        """联邦训练模式（完整模型训练）并生成验证证明"""
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

        # 生成验证证明
        model.eval()
        with torch.no_grad():
            challenge_outputs = model(self.challenge_data)
        proof = challenge_outputs.cpu().numpy()
        return model.state_dict(), proof

    def _split_train(self, ue_model, server_model, user_idx):
        """分割训练模式并生成验证证明"""
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

        # 生成验证证明
        ue_model.eval()
        server_model.eval()
        with torch.no_grad():
            ue_output = ue_model.forward_ue(self.challenge_data)
            server_output = server_model.forward_server(ue_output)
        proof = server_output.cpu().numpy()
        return ue_model.ue_layers.state_dict(), proof

    def _verify_proof(self, model_weights, proof, is_split):
        """验证证明有效性"""
        if is_split:
            # 分割训练验证
            ue_model = SplitModel(self._init_base_model(type(self.base_model).__name__))
            ue_model.ue_layers.load_state_dict(model_weights)
            ue_model.to(self.device)
            server_model = SplitModel(self._init_base_model(type(self.base_model).__name__))
            server_model.server_layers.load_state_dict(self.global_model.server_layers.state_dict())
            server_model.to(self.device)
            ue_model.eval()
            server_model.eval()
            with torch.no_grad():
                ue_output = ue_model.forward_ue(self.challenge_data)
                server_output = server_model.forward_server(ue_output)
            computed_proof = server_output.cpu().numpy()
        else:
            # 联邦训练验证
            model = self._init_base_model(type(self.base_model).__name__)
            model.load_state_dict(model_weights)
            model.to(self.device)
            model.eval()
            with torch.no_grad():
                outputs = model(self.challenge_data)
            computed_proof = outputs.cpu().numpy()

        # 计算均方误差
        mse = np.mean((computed_proof - proof) ** 2)
        return mse <= self.proof_threshold

    def aggregate(self, fed_weights, split_weights):
        """混合聚合策略（仅聚合通过验证的参数）"""
        global_ue = self.global_model.ue_layers.state_dict()
        global_server = self.global_model.server_layers.state_dict()

        # 处理联邦训练更新
        fed_ue_updates = {key: [] for key in global_ue.keys()}
        fed_server_updates = {key: [] for key in global_server.keys()}
        if fed_weights:
            for full_weights in fed_weights:
                # 分解参数到UE和Server部分
                for key in global_ue:
                    base_key = key.replace('ue_layers.', '', 1)
                    if base_key in full_weights:
                        fed_ue_updates[key].append(full_weights[base_key])
                for key in global_server:
                    base_key = key.replace('server_layers.', '', 1)
                    if base_key in full_weights:
                        fed_server_updates[key].append(full_weights[base_key])

        # 处理分割训练更新
        split_ue_updates = {key: [] for key in global_ue.keys()}
        if split_weights:
            for ue_weights in split_weights:
                for key in global_ue:
                    if key in ue_weights:
                        split_ue_updates[key].append(ue_weights[key])

        # 聚合UE部分
        for key in global_ue:
            updates = []
            if key in fed_ue_updates and fed_ue_updates[key]:
                updates.extend(fed_ue_updates[key])
            if key in split_ue_updates and split_ue_updates[key]:
                updates.extend(split_ue_updates[key])
            if updates:
                global_ue[key] = torch.mean(torch.stack(updates), dim=0)

        # 聚合Server部分
        for key in global_server:
            updates = []
            if key in fed_server_updates and fed_server_updates[key]:
                updates.extend(fed_server_updates[key])
            if updates:
                global_server[key] = torch.mean(torch.stack(updates), dim=0)

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
        """执行一轮HSFL训练（含可验证机制）"""
        fed_weights, fed_proofs = [], []
        split_weights, split_proofs = [], []

        # 执行本地训练
        for user in selected_users:
            if user in split_users:
                # 分割训练
                ue_model = SplitModel(self._init_base_model(type(self.base_model).__name__))
                ue_model.ue_layers.load_state_dict(self.global_model.ue_layers.state_dict())
                server_model = SplitModel(self._init_base_model(type(self.base_model).__name__))
                server_model.server_layers.load_state_dict(self.global_model.server_layers.state_dict())
                ue_weights, proof = self._split_train(ue_model, server_model, user)
                split_weights.append(ue_weights)
                split_proofs.append(proof)
            else:
                # 联邦训练
                full_model = self._init_base_model(type(self.base_model).__name__)
                # 转换参数格式
                state_dict = {}
                for k, v in self.global_model.ue_layers.state_dict().items():
                    key = k.replace("ue_layers.", "", 1)
                    state_dict[key] = v
                for k, v in self.global_model.server_layers.state_dict().items():
                    key = k.replace("server_layers.", "", 1)
                    state_dict[key] = v
                full_model.load_state_dict(state_dict)
                full_weights, proof = self._federated_train(full_model, user)
                fed_weights.append(full_weights)
                fed_proofs.append(proof)

        # 参数验证
        valid_fed = [w for w, p in zip(fed_weights, fed_proofs) if self._verify_proof(w, p, False)]
        valid_split = [w for w, p in zip(split_weights, split_proofs) if self._verify_proof(w, p, True)]

        # 聚合有效参数
        self.aggregate(valid_fed, valid_split)
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
        split_ratio=0.5,
        num_challenges=10,
        proof_threshold=1e-4
    )

    # 模拟训练过程
    for round in range(10):
        selected = np.random.choice(100, size=10, replace=False)
        split_users = np.random.choice(selected, size=int(10 * 0.5), replace=False)
        acc = hsfl.train_round(selected, split_users)
        print(f"Round {round + 1}, Test Accuracy: {acc:.4f}")
        # print(f"Split Layer Shape: {hsfl.split_layer_shape}")