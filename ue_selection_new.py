# ue_selection_new.py
import numpy as np
import torch
from sklearn.cluster import KMeans
import itertools



class MABSelector:
    def __init__(self, num_users, beta=0.5, lambda_l=0.9, lambda_c=0.9,
                 sigma=1.0, init_omega=1.0, init_gamma=1.0,
                 # 博弈参数
                 alpha=0.5, beta_g=0.7, gamma_shapley=0.3, gamma_penalty=0.95,
                 delta_historical=0.2, coalition_period=10,
                 min_coalition_size=3, max_coalition_size=10,
                 shapley_samples=1000, coalition_variance_threshold=0.3):
        """
        多臂老虎机用户选择器（MAB-BC-BN2策略）与博弈论结合

        新增博弈参数：
            alpha: 能耗系数（用于效用函数）
            beta_g: 非合作博弈中的权重参数
            gamma_shapley: 联盟贡献的权重系数
            gamma_penalty: 惩罚机制的衰减因子
            delta_historical: 历史贡献权重
            coalition_period: 联盟更新周期
            min_coalition_size: 最小联盟大小
            max_coalition_size: 最大联盟大小
            shapley_samples: Shapley值计算的蒙特卡洛样本数
            coalition_variance_threshold: 联盟拆分的方差阈值
        """
        self.num_users = num_users
        self.beta = beta
        self.lambda_l = lambda_l
        self.lambda_c = lambda_c
        self.sigma = sigma
        self.alpha = alpha
        self.beta_g = beta_g
        self.gamma_shapley = gamma_shapley
        self.gamma_penalty = gamma_penalty
        self.delta_historical = delta_historical
        self.coalition_period = coalition_period
        self.min_coalition_size = min_coalition_size
        self.max_coalition_size = max_coalition_size
        self.shapley_samples = shapley_samples
        self.coalition_variance_threshold = coalition_variance_threshold

        # 初始化历史记录
        self.omega = np.full(num_users, init_omega)  # 模型更新历史（BN2）
        self.gamma = np.full(num_users, init_gamma)  # 信道质量历史（BC）
        self.counts = np.zeros(num_users)  # 选择次数计数器
        self.t = 0  # 总时间步

        # 非合作博弈相关
        self.last_delta_omegas = np.zeros(num_users)  # 上一轮的模型更新量
        self.candidate_pool = np.arange(num_users)  # 候选设备池
        self.energy_costs = np.random.rand(num_users) * 0.5 + 0.5  # 模拟能耗成本 (0.5-1.0)

        # 合作博弈相关
        self.coalitions = []  # 设备联盟列表
        self.shapley_values = np.zeros(num_users)  # Shapley值
        self.last_coalition_update = 0  # 上次联盟更新时间

        # 重复博弈相关
        self.historical_contributions = []  # 历史贡献记录
        self.reject_counts = np.zeros(num_users)  # 拒绝参与次数
        self.trust_factors = np.ones(num_users)  # 信任因子
        self.participation_history = np.zeros(num_users)  # 参与历史记录

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

        # 保存当前模型更新量用于下一轮的非合作博弈
        self.last_delta_omegas = delta_omegas.copy()

        # 对每个用户应用折扣因子
        self.omega *= self.lambda_l
        self.gamma *= self.lambda_c
        self.counts *= self.lambda_l  # 假设使用相同折扣因子

        # 更新被选用户的信息
        for u in selected_users:
            u = int(u)  # 确保索引为整数
            # 更新模型更新历史（折扣累加）
            self.omega[u] += (1 - self.lambda_l) * delta_omegas[u]
            # 更新信道质量历史（折扣累加）
            self.gamma[u] += (1 - self.lambda_c) * snrs[u]
            # 更新选择次数（折扣累加）
            self.counts[u] += (1 - self.lambda_l)
            # 更新参与历史
            self.participation_history[u] += 1

            # 更新信任因子（奖励参与）
            self.trust_factors[u] = min(1.0, self.trust_factors[u] + 0.05)

        # 更新拒绝参与设备的信任因子（惩罚） - 只惩罚明确拒绝的设备
        all_users = np.arange(self.num_users)
        non_participating = np.setdiff1d(all_users, self.candidate_pool)
        for u in non_participating:
            self.reject_counts[u] += 1
            self.trust_factors[u] = max(0.1, self.trust_factors[u] * self.gamma_penalty)

        # 记录历史贡献（用于重复博弈）
        historical_contribution = np.zeros(self.num_users)
        for u in selected_users:
            u = int(u)
            # 综合贡献 = SNR + 模型更新量
            historical_contribution[u] = snrs[u] + delta_omegas[u]
        self.historical_contributions.append(historical_contribution)
        if len(self.historical_contributions) > 100:  # 保留最近100轮
            self.historical_contributions.pop(0)

        # 定期更新联盟（合作博弈）
        if self.t - self.last_coalition_update >= self.coalition_period:
            self.update_coalitions(snrs)
            self.last_coalition_update = self.t

    def calculate_utility(self, snrs):
        """计算每个设备的效用函数值（修正版）"""
        # 效用函数: U_n = beta_g * SNR_n + (1 - beta_g) * ||Δω_n||_2 - alpha * Energy_n
        # 其中Energy_n是能耗成本（与模型更新量相关）
        energy_cost = self.alpha * self.energy_costs * self.last_delta_omegas
        return (self.beta_g * snrs +
                (1 - self.beta_g) * self.last_delta_omegas -
                energy_cost)

    def self_selection(self, snrs):
        """非合作博弈：设备自选择是否参与"""
        utility = self.calculate_utility(snrs)
        # 效用>=0的设备愿意参与
        candidate_mask = utility >= 0
        self.candidate_pool = np.where(candidate_mask)[0]
        return self.candidate_pool

    def calculate_shapley_value(self, coalition, snrs):
        """计算联盟内各设备的Shapley值（蒙特卡洛近似）"""
        n = len(coalition)
        shapley_values = np.zeros(n)
        device_to_index = {device: idx for idx, device in enumerate(coalition)}

        # 使用蒙特卡洛方法近似计算Shapley值
        for _ in range(self.shapley_samples):
            # 随机排列联盟中的设备
            perm = np.random.permutation(coalition)
            # 计算累积价值
            cumulative_value = 0.0
            for i, device in enumerate(perm):
                # 当前子集：排列中当前设备之前的所有设备
                subset = perm[:i]
                # 子集S的价值
                S_val = self.coalition_value(subset, snrs)
                # 子集S ∪ {i}的价值
                S_i_val = self.coalition_value(np.append(subset, device), snrs)
                # 边际贡献
                marginal = S_i_val - S_val
                # 累加到对应设备的Shapley值
                idx = device_to_index[device]
                shapley_values[idx] += marginal

        # 平均边际贡献
        shapley_values /= self.shapley_samples

        return shapley_values

    def coalition_value(self, coalition, snrs):
        """计算联盟的价值函数"""
        if len(coalition) == 0:
            return 0.0

        # v(S) = sum_{i in S} (SNR_i + ||Δω_i||_2)
        snr_sum = np.sum([snrs[i] for i in coalition])
        delta_sum = np.sum([self.last_delta_omegas[i] for i in coalition])
        return snr_sum + delta_sum

    def update_coalitions(self, snrs):
        """更新设备联盟（合作博弈）包含合并和拆分逻辑"""
        # 使用K-means聚类（基于SNR和模型更新量）
        features = np.column_stack((
            (snrs - np.mean(snrs)) / np.std(snrs),
            (self.last_delta_omegas - np.mean(self.last_delta_omegas)) / np.std(self.last_delta_omegas)
        ))

        # 确定联盟数量（基于设备总数）
        n_coalitions = max(2, min(
            self.max_coalition_size,
            self.num_users // self.min_coalition_size
        ))

        kmeans = KMeans(n_clusters=n_coalitions, random_state=0).fit(features)
        labels = kmeans.labels_

        # 创建初始联盟
        new_coalitions = []
        for i in range(n_coalitions):
            coalition = np.where(labels == i)[0]
            if len(coalition) > 0:
                new_coalitions.append(coalition.tolist())

        # 计算每个联盟的Shapley值并检查是否需要拆分
        final_coalitions = []
        self.shapley_values = np.zeros(self.num_users)

        for coalition in new_coalitions:
            if len(coalition) < 2:  # 单设备联盟不需要拆分
                final_coalitions.append(coalition)
                continue

            # 计算联盟内各设备的Shapley值
            shapley = self.calculate_shapley_value(coalition, snrs)

            # 检查Shapley值方差是否超过阈值
            if np.var(shapley) > self.coalition_variance_threshold:
                # 方差过大，拆分联盟
                sorted_indices = np.argsort(shapley)[::-1]  # 按贡献降序排列
                split_point = len(coalition) // 2
                high_contrib = [coalition[i] for i in sorted_indices[:split_point]]
                low_contrib = [coalition[i] for i in sorted_indices[split_point:]]

                if len(high_contrib) > 0:
                    final_coalitions.append(high_contrib)
                if len(low_contrib) > 0:
                    final_coalitions.append(low_contrib)
            else:
                # 保留原联盟
                final_coalitions.append(coalition)

        # 合并小型联盟（文档要求）
        merged_coalitions = []
        small_coalitions = [c for c in final_coalitions if len(c) < self.min_coalition_size]
        large_coalitions = [c for c in final_coalitions if len(c) >= self.min_coalition_size]

        # 合并所有小型联盟为一个
        if small_coalitions:
            merged_small = list(itertools.chain.from_iterable(small_coalitions))
            if len(merged_small) > 0:
                merged_coalitions.append(merged_small)

        # 添加大型联盟
        merged_coalitions.extend(large_coalitions)

        # 更新联盟
        self.coalitions = merged_coalitions

        # 重新计算所有联盟的Shapley值
        self.shapley_values = np.zeros(self.num_users)
        for coalition in self.coalitions:
            if len(coalition) > 0:
                shapley = self.calculate_shapley_value(coalition, snrs)
                for idx, device in enumerate(coalition):
                    self.shapley_values[device] = shapley[idx]

        # 标准化Shapley值
        max_shapley = np.max(self.shapley_values)
        if max_shapley > 0:
            self.shapley_values /= max_shapley

    def calculate_historical_contribution(self, user_idx):
        """计算用户的历史贡献（重复博弈）"""
        if not self.historical_contributions:
            return 0.0

        # 计算最近T轮的平均贡献
        window_size = min(10, len(self.historical_contributions))
        recent_contributions = [hc[user_idx] for hc in self.historical_contributions[-window_size:]]
        return np.mean(recent_contributions)

    def calculate_ucb_scores(self, snrs):
        """计算所有用户的UCB分数（集成博弈策略）"""
        # 计算标准化后的指标
        norm_omega = (self.omega - np.min(self.omega)) / (
                np.max(self.omega) - np.min(self.omega) + 1e-6)
        norm_gamma = (self.gamma - np.min(self.gamma)) / (
                np.max(self.gamma) - np.min(self.gamma) + 1e-6)

        # 计算探索项
        total_counts = np.sum(self.counts)
        exploration = self.sigma * np.sqrt(
            2 * np.log(total_counts + 1) / (self.counts + 1e-6))

        # 基础UCB分数
        scores = (self.beta * norm_omega +
                  (1 - self.beta) * norm_gamma +
                  exploration)

        # 加入联盟贡献（合作博弈）
        scores += self.gamma_shapley * self.shapley_values

        # 加入历史贡献（重复博弈）
        historical_contrib = np.array([self.calculate_historical_contribution(u)
                                       for u in range(self.num_users)])
        if np.max(historical_contrib) > 0:
            norm_historical = historical_contrib / np.max(historical_contrib)
            scores += self.delta_historical * norm_historical

        # 应用信任因子（惩罚机制）
        scores *= self.trust_factors

        # 加入长期参与奖励
        participation_ratio = self.participation_history / (self.t + 1e-6)
        scores += 0.1 * participation_ratio  # 奖励长期参与者

        return scores

    def select_users(self, K, snrs, return_scores=False):
        """
        选择Top-K用户（集成博弈策略）
        步骤：
          1. 非合作博弈：设备自选择形成候选池
          2. 在候选池上计算UCB分数
          3. 选择Top-K用户
        """
        # 1. 非合作博弈：设备自选择
        self.self_selection(snrs)

        # 2. 计算候选池中设备的UCB分数
        scores = self.calculate_ucb_scores(snrs)

        # 将非候选设备的分数设为负无穷（确保不会被选择）
        non_candidate_mask = np.ones(self.num_users, dtype=bool)
        non_candidate_mask[self.candidate_pool] = False
        scores[non_candidate_mask] = -np.inf

        # 3. 选择Top-K用户
        selected = np.argsort(-scores)[:K]  # 降序排列取前K

        return (selected, scores) if 0 else selected

    def reset(self):
        """重置所有历史记录"""
        self.omega.fill(1.0)
        self.gamma.fill(1.0)
        self.counts.fill(0)
        self.t = 0
        self.last_delta_omegas.fill(0)
        self.candidate_pool = np.arange(self.num_users)
        self.coalitions = []
        self.shapley_values.fill(0)
        self.last_coalition_update = 0
        self.historical_contributions = []
        self.reject_counts.fill(0)
        self.trust_factors.fill(1.0)
        self.participation_history.fill(0)
        self.energy_costs = np.random.rand(self.num_users) * 0.5 + 0.5  # 重置能耗成本


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
            # 使用更稳定的L2范数计算
            if isinstance(global_model[key], torch.Tensor):
                diff = local[key] - global_model[key]
                delta += torch.sum(diff ** 2).item()
            else:
                # 处理非Tensor类型
                diff = local[key] - global_model[key]
                delta += np.sum(np.square(diff))
        delta_norms.append(np.sqrt(delta))
    return np.array(delta_norms)


# 测试代码
if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    # 初始化选择器（100用户），添加博弈参数
    selector = MABSelector(
        num_users=100,
        beta=0.5,
        alpha=0.3,  # 能耗系数
        beta_g=0.6,  # 非合作博弈权重
        gamma_shapley=0.4,  # 联盟贡献权重
        gamma_penalty=0.9,  # 惩罚衰减因子
        delta_historical=0.3,  # 历史贡献权重
        coalition_period=5,  # 每5轮更新联盟
        shapley_samples=500,  # Shapley值计算样本数
        coalition_variance_threshold=0.2  # 联盟拆分阈值
    )

    # 模拟200轮选择
    for round_idx in range(200):
        # 随机生成模拟数据
        snrs = np.random.normal(20, 5, 100)  # 生成SNR
        global_model = {'weight': torch.randn(10)}  # 模拟全局模型
        local_models = [{'weight': torch.randn(10)} for _ in range(100)]  # 模拟本地模型

        # 计算模型更新量
        delta_norms = calculate_delta_norms(global_model, local_models)

        # 选择用户（假设选择10个）
        selected = selector.select_users(10, snrs)

        # 更新选择器状态
        selector.update(selected, delta_norms, snrs)

        # 打印选择结果
        print(f"Round {round_idx + 1}:")
        print(f"  Selected users: {selected}")
        print(f"  Candidate pool size: {len(selector.candidate_pool)}")
        print(f"  Avg trust factor: {np.mean(selector.trust_factors):.2f}")
        print(f"  Non-participating devices: {100 - len(selector.candidate_pool)}")

        # 每10轮打印联盟信息
        if (round_idx + 1) % 10 == 0:
            print("\nCoalition Info:")
            for i, coalition in enumerate(selector.coalitions):
                shapley_in_coalition = [selector.shapley_values[d] for d in coalition]
                variance = np.var(shapley_in_coalition) if len(coalition) > 1 else 0
                print(f"  Coalition {i + 1} (size={len(coalition)}, var={variance:.4f}): {coalition}")
            print(f"  Max Shapley value: {np.max(selector.shapley_values):.4f}")
            print(f"  Min Shapley value: {np.min(selector.shapley_values):.4f}\n")