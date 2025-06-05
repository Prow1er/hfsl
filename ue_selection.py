import numpy as np
import torch


class MABSelector:
    def __init__(self, num_users, beta=0.5, lambda_l=0.9, lambda_c=0.9,
                 sigma=1.0, init_omega=1.0, init_gamma=1.0):
        """
        多臂老虎机用户选择器（MAB-BC-BN2策略）

        参数：
            num_users: 用户总数
            beta: 模型更新与信道质量的平衡因子（0-1）
            lambda_l: 模型更新的折扣因子（0-1）
            lambda_c: 信道质量的折扣因子（0-1）
            sigma: 探索强度参数
            init_omega: 模型更新的初始值
            init_gamma: 信道质量的初始值
        """
        self.num_users = num_users
        self.beta = beta
        self.lambda_l = lambda_l
        self.lambda_c = lambda_c
        self.sigma = sigma

        # 初始化历史记录
        self.omega = np.full(num_users, init_omega)  # 模型更新历史（BN2）
        self.gamma = np.full(num_users, init_gamma)  # 信道质量历史（BC）
        self.counts = np.zeros(num_users)  # 选择次数计数器
        self.t = 0  # 总时间步

    def update(self, selected_users, delta_omegas, snrs):
        """
        更新用户历史记录

        参数：
            selected_users: 本轮选择的用户索引列表
            delta_omegas: 各用户的模型更新L2范数（Tensor或np.array）
            snrs: 各用户的当前SNR测量值（np.array）
        """
        # 转换输入数据格式
        if isinstance(delta_omegas, torch.Tensor):
            delta_omegas = delta_omegas.cpu().numpy()
        delta_omegas = np.abs(delta_omegas)  # 确保值为正

        # 更新全局时间步
        self.t += 1

        # 对每个用户应用折扣因子
        self.omega *= self.lambda_l
        self.gamma *= self.lambda_c
        self.counts *= self.lambda_l  # 假设使用相同折扣因子（论文未明确）

        # 更新被选用户的信息
        for u in selected_users:
            u = int(u)  # 确保索引为整数
            # 更新模型更新历史（折扣累加）
            self.omega[u] += (1 - self.lambda_l) * delta_omegas[u]
            # 更新信道质量历史（折扣累加）
            self.gamma[u] += (1 - self.lambda_c) * snrs[u]
            # 更新选择次数（折扣累加）
            self.counts[u] += (1 - self.lambda_l)

    def calculate_ucb_scores(self):
        """计算所有用户的UCB分数"""
        # 计算标准化后的指标
        norm_omega = (self.omega - np.min(self.omega)) / (
                np.max(self.omega) - np.min(self.omega) + 1e-6)
        norm_gamma = (self.gamma - np.min(self.gamma)) / (
                np.max(self.gamma) - np.min(self.gamma) + 1e-6)

        # 计算探索项
        total_counts = np.sum(self.counts)
        exploration = self.sigma * np.sqrt(
            2 * np.log(total_counts + 1) / (self.counts + 1e-6))

        # 组合UCB分数
        scores = (self.beta * norm_omega +
                  (1 - self.beta) * norm_gamma +
                  exploration)
        return scores

    def select_users(self, K, return_scores=False):
        """
        选择Top-K用户

        返回：
            selected_users: 选择的用户索引（按分数降序）
            scores: 所有用户的分数（如果return_scores=True）
        """
        scores = self.calculate_ucb_scores()
        selected = np.argsort(-scores)[:K]  # 降序排列取前K
        return (selected, scores) if return_scores else selected

    def reset(self):
        """重置所有历史记录"""
        self.omega.fill(1.0)
        self.gamma.fill(1.0)
        self.counts.fill(0)
        self.t = 0


def calculate_delta_norms(global_model, local_models):
    """
    计算各本地模型与全局模型的L2范数差异

    参数：
        global_model: 当前全局模型状态字典
        local_models: 各本地模型状态字典列表

    返回：
        delta_norms: 各用户的模型更新L2范数（np.array）
    """
    delta_norms = []
    for local in local_models:
        delta = 0.0
        for key in global_model:
            delta += torch.sum(
                (local[key] - global_model[key]) ** 2).item()
        delta_norms.append(np.sqrt(delta))
    return np.array(delta_norms)


if __name__ == "__main__":
    np.random.seed(42)

    # 初始化选择器（100用户，beta=0.5）
    selector = MABSelector(num_users=100, beta=0.5)

    # 模拟10轮选择
    for round in range(200):
        # 随机生成模拟数据
        snrs = np.random.normal(20, 5, 100)  # 生成SNR
        global_model = {'weight': torch.randn(10)}  # 模拟全局模型
        local_models = [{'weight': torch.randn(10)} for _ in range(100)]  # 模拟本地模型

        # 计算模型更新量（示例）
        delta_norms = calculate_delta_norms(global_model, local_models)

        # 选择用户（假设选择10个）
        selected = selector.select_users(10)

        # 更新选择器状态
        selector.update(selected, delta_norms, snrs)

        # 打印选择结果
        print(f"Round {round + 1} selected users:")
        print("Indices:", selected)
        print("SNRs:", snrs[selected].round(1))
        print("Norms:", delta_norms[selected].round(2))
        print("Current counts:", selector.counts[selected].round(2))
        print("---")