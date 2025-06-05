# experiment_hsfvl.py
import numpy as np
import torch
import matplotlib.pyplot as plt
from data_utils import load_mnist, noniid_partition, iid_partition
from hsfl import HSFLTrainer as HSFL
from hsfl_new import HSFLTrainer as HSVFL
import time
import copy
from models import SplitModel


class MaliciousHSFLTrainer(HSFL):
    """扩展HSFL以支持恶意攻击"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attacked_users = set()
        self.malicious_updates = {}

    def train_round(self, selected_users, split_users):
        """重写训练轮次以支持恶意攻击"""
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

                # 如果是恶意用户，存储恶意更新
                if user in self.attacked_users:
                    self.malicious_updates[user] = copy.deepcopy(ue_weights)
                    # 模型替换攻击：使用恶意模型
                    for key in ue_weights:
                        ue_weights[key] = torch.randn_like(ue_weights[key]) * 5.0

                split_weights.append(ue_weights)
            else:
                # 联邦训练用户
                full_model = self._init_base_model(type(self.base_model).__name__)
                # 将 SplitModel 的参数还原为基础模型格式
                state_dict = {}
                for k, v in self.global_model.ue_layers.state_dict().items():
                    key = k.replace("ue_layers.", "", 1)
                    state_dict[key] = v
                for k, v in self.global_model.server_layers.state_dict().items():
                    key = k.replace("server_layers.", "", 1)
                    state_dict[key] = v
                full_model.load_state_dict(state_dict)
                full_weights = self._federated_train(full_model, user)

                # 如果是恶意用户，存储恶意更新
                if user in self.attacked_users:
                    self.malicious_updates[user] = copy.deepcopy(full_weights)
                    # 模型替换攻击：使用恶意模型
                    for key in full_weights:
                        full_weights[key] = torch.randn_like(full_weights[key]) * 5.0

                fed_weights.append(full_weights)

        # 模型聚合
        self.aggregate(fed_weights, split_weights)
        return self.test()


class MaliciousHSVFLTrainer(HSVFL):
    """扩展HSVFL以支持恶意攻击"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attacked_users = set()
        self.malicious_updates = {}

    def train_round(self, selected_users, split_users):
        """重写训练轮次以支持恶意攻击"""
        fed_weights, fed_proofs = [], []
        split_weights, split_proofs = [], []

        for user in selected_users:
            if user in split_users:
                # 分割训练
                ue_model = SplitModel(self._init_base_model(type(self.base_model).__name__))
                ue_model.ue_layers.load_state_dict(self.global_model.ue_layers.state_dict())
                server_model = SplitModel(self._init_base_model(type(self.base_model).__name__))
                server_model.server_layers.load_state_dict(self.global_model.server_layers.state_dict())
                ue_weights, proof = self._split_train(ue_model, server_model, user)

                # 如果是恶意用户，存储恶意更新
                if user in self.attacked_users:
                    self.malicious_updates[user] = copy.deepcopy(ue_weights)
                    # 模型替换攻击：使用恶意模型
                    for key in ue_weights:
                        ue_weights[key] = torch.randn_like(ue_weights[key]) * 5.0

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

                # 如果是恶意用户，存储恶意更新
                if user in self.attacked_users:
                    self.malicious_updates[user] = copy.deepcopy(full_weights)
                    # 模型替换攻击：使用恶意模型
                    for key in full_weights:
                        full_weights[key] = torch.randn_like(full_weights[key]) * 5.0

                fed_weights.append(full_weights)
                fed_proofs.append(proof)

        # 参数验证
        valid_fed = [w for w, p in zip(fed_weights, fed_proofs) if self._verify_proof(w, p, False)]
        valid_split = [w for w, p in zip(split_weights, split_proofs) if self._verify_proof(w, p, True)]

        # 计算检测率
        total_malicious = sum(1 for user in selected_users if user in self.attacked_users)
        detected_malicious = total_malicious - len([w for w, p in zip(fed_weights, fed_proofs)
                                                    if self._verify_proof(w, p, False) and
                                                    any(user in self.attacked_users for user in selected_users)])
        detection_rate = detected_malicious / total_malicious if total_malicious > 0 else 0.0

        # 聚合有效参数
        self.aggregate(valid_fed, valid_split)
        return self.test(), detection_rate


def run_experiment(attack_ratio=0.0, num_rounds=100, num_users=100, selected_per_round=10):
    """运行对比实验 100ues Net iid k=10"""
    torch.manual_seed(42)
    np.random.seed(42)

    train_set, test_set = load_mnist()
    train_set.partitions = iid_partition(train_set, num_users=num_users)

    hsfl_trainer = MaliciousHSFLTrainer(
        num_users=num_users,
        train_set=train_set,
        test_set=test_set,
        model_type='Net',
        split_ratio=0.5,
        lr=0.01,
        batch_size=32,
        local_epochs=3
    )

    hsvfl_trainer = MaliciousHSVFLTrainer(
        num_users=num_users,
        train_set=train_set,
        test_set=test_set,
        model_type='Net',
        split_ratio=0.5,
        lr=0.01,
        batch_size=32,
        local_epochs=3,
        num_challenges=10,
        proof_threshold=1e-4
    )

    # 创建恶意用户（固定）
    malicious_users = set(np.random.choice(
        num_users,
        size=int(num_users * attack_ratio),
        replace=False
    ))

    # 设置恶意用户
    hsfl_trainer.attacked_users = malicious_users
    hsvfl_trainer.attacked_users = malicious_users

    results = {
        'HSFL_acc': [],
        'HSVFL_acc': [],
        'HSFL_time': [],
        'HSVFL_time': [],
        'HSFL_comm': [],
        'HSVFL_comm': [],
        'HSVFL_detection': []
    }

    split_layer_size = np.prod(hsvfl_trainer.split_layer_shape)
    full_model_size = sum(p.numel() for p in hsfl_trainer.global_model.parameters())

    for round in range(num_rounds):
        selected = np.random.choice(num_users, size=selected_per_round, replace=False)
        split_users = np.random.choice(
            selected,
            size=int(selected_per_round * 0.5),
            replace=False
        )

        # 运行HSFL
        start_time = time.time()
        hsfl_acc = hsfl_trainer.train_round(selected, split_users)
        hsfl_time = time.time() - start_time

        # 运行HSVFL
        start_time = time.time()
        hsvfl_acc, detection_rate = hsvfl_trainer.train_round(selected, split_users)
        hsvfl_time = time.time() - start_time

        results['HSFL_acc'].append(hsfl_acc)
        results['HSVFL_acc'].append(hsvfl_acc)
        results['HSFL_time'].append(hsfl_time)
        results['HSVFL_time'].append(hsvfl_time)
        results['HSVFL_detection'].append(detection_rate)

        hsfl_comm = 0
        hsvfl_comm = 0

        for user in selected:
            if user in split_users:
                hsfl_comm += split_layer_size
                hsvfl_comm += split_layer_size + 10 * 10
            else:
                hsfl_comm += full_model_size
                hsvfl_comm += full_model_size + 10 * 10

        results['HSFL_comm'].append(hsfl_comm)
        results['HSVFL_comm'].append(hsvfl_comm)

        print(f"Round {round + 1}/{num_rounds} | "
              f"HSFL: {hsfl_acc:.4f} | "
              f"HSVFL: {hsvfl_acc:.4f} | "
              f"Detection Rate: {detection_rate:.2f} | "
              f"Malicious: {len(malicious_users)}")

    return results


def plot_results(results, attack_ratio):
    """可视化实验结果"""
    plt.figure(figsize=(32, 24))
    plt.rcParams['font.size'] = 20
    plt.rcParams['axes.titlesize'] = 22
    plt.rcParams['axes.labelsize'] = 20
    plt.subplots_adjust(left=0.07, right=0.93, bottom=0.1, top=0.95, hspace=0.35, wspace=0.15)

    # 准确率对比
    ax1 = plt.subplot(2, 2, 1)
    plt.plot(results['HSFL_acc'], 'b-', label='HSFL', marker='o', markersize=8)
    plt.plot(results['HSVFL_acc'], 'r-', label='HSVFL', marker='x', markersize=8)
    ax1.set_title(f'(a) Test Accuracy (Attack Ratio: {attack_ratio * 100}%)', fontsize=22, y=-0.15)
    ax1.set_xlabel('Communication Round')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    ax1.set_ylim(0, 1)

    # # 时间开销对比
    # ax3 = plt.subplot(3, 2, 2)
    # plt.plot(results['HSFL_time'], 'b-', label='HSFL', marker='o', markersize=8)
    # plt.plot(results['HSVFL_time'], 'r-', label='HSVFL', marker='x', markersize=8)
    # ax3.set_title('(b) Computation Time per Round', fontsize=22)
    # ax3.set_xlabel('Communication Round')
    # ax3.set_ylabel('Time (seconds)')
    # ax3.legend()
    # ax3.grid(True)
    #
    # # 通信开销对比
    # ax2 = plt.subplot(3, 2, 3)
    # plt.plot(np.cumsum(results['HSFL_comm']), 'b-', label='HSFL')
    # plt.plot(np.cumsum(results['HSVFL_comm']), 'r-', label='HSVFL')
    # ax2.set_title('(c) Cumulative Communication Cost (Bytes)', fontsize=22)
    # ax2.set_xlabel('Communication Round')
    # ax2.set_ylabel('Total Bytes')
    # ax2.legend()
    # ax2.grid(True)

    # 安全性能展示
    ax4 = plt.subplot(2, 2, 2)
    if attack_ratio > 0:
        plt.plot(results['HSVFL_detection'], 'g-', label='Malicious Update Detection Rate', linewidth=3)
        plt.axhline(y=np.mean(results['HSVFL_detection']), color='m', linestyle='--',
                    label=f'Avg: {np.mean(results["HSVFL_detection"]):.2f}')
        plt.ylim(0, 1)
        ax4.set_title('(b) Security Performance', fontsize=22, y=-0.15)
        ax4.set_xlabel('Communication Round')
        ax4.set_ylabel('Detection Rate')
        ax4.legend()
        ax4.grid(True)

    # 恶意攻击影响对比
    ax5 = plt.subplot(2, 2, 3)
    hsfl_drop = [max(results['HSFL_acc']) - acc for acc in results['HSFL_acc']]
    hsvfl_drop = [max(results['HSVFL_acc']) - acc for acc in results['HSVFL_acc']]
    plt.plot(hsfl_drop, 'b-', label='HSFL Accuracy Drop', marker='o', markersize=8)
    plt.plot(hsvfl_drop, 'r-', label='HSVFL Accuracy Drop', marker='x', markersize=8)
    ax5.set_title('(c) Accuracy Drop Due to Attacks', fontsize=22, y=-0.15)
    ax5.set_xlabel('Communication Round')
    ax5.set_ylabel('Accuracy Drop')
    ax5.legend()
    ax5.grid(True)

    # 检测率与准确率关系
    ax6 = plt.subplot(2, 2, 4)
    if attack_ratio > 0:
        scatter = ax6.scatter(results['HSVFL_detection'], results['HSVFL_acc'],
                              c=range(len(results['HSVFL_acc'])), cmap='viridis', s=100)
        ax6.set_title('(d) Detection Rate vs Accuracy', fontsize=22, y=-0.15)
        ax6.set_xlabel('Detection Rate')
        ax6.set_ylabel('Accuracy')
        ax6.grid(True)
        plt.colorbar(scatter, ax=ax6, label='Round Index')

    plt.tight_layout()
    plt.savefig(f'comparison_attack_{attack_ratio}.png', dpi=300)
    plt.show()


def main():
    """主实验函数"""

    # 实验2: 5%恶意用户
    print("\nRunning experiment with 0% malicious users...")
    results_10p_attack = run_experiment(attack_ratio=0.05, num_rounds=100)
    plot_results(results_10p_attack, 0.05)

    # # 实验3: 30%恶意用户
    # print("\nRunning experiment with 30% malicious users...")
    # results_30p_attack = run_experiment(attack_ratio=0.3, num_rounds=100)
    # plot_results(results_30p_attack, 0.3)

    # # 实验4: 80%恶意用户
    # print("\nRunning experiment with 80% malicious users...")
    # results_80p_attack = run_experiment(attack_ratio=0.8, num_rounds=100)
    # plot_results(results_80p_attack, 0.8)


if __name__ == "__main__":
    main()