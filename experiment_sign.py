import numpy as np
import matplotlib.pyplot as plt
from secure_selector import PKIManager, SecureMABSelector


class SignatureVerificationExperiment:
    def __init__(self, num_users=100, malicious_ratio=0.2, num_rounds=50):
        """
        数字签名验证实验
        参数:
            num_users: 总用户数
            malicious_ratio: 恶意用户比例
            num_rounds: 实验轮数
        """
        self.num_users = num_users
        self.malicious_ratio = malicious_ratio
        self.num_rounds = num_rounds
        self.num_malicious = int(num_users * malicious_ratio)

        # 初始化PKI和选择器
        self.pki = PKIManager(num_users)
        self.selector = SecureMABSelector(
            pki_manager=self.pki,
            num_users=num_users,
            beta=0.5,
            alpha=0.3,
            beta_g=0.6,
            gamma_shapley=0.4,
            gamma_penalty=0.9,
            delta_historical=0.3,
            coalition_period=5,
            shapley_samples=500,
            coalition_variance_threshold=0.2
        )

        # 实验数据记录
        self.results = {
            'detection_accuracy': [],
            'false_positive_rate': [],
            'false_negative_rate': [],
            'trust_score_diff': [],
            'malicious_behavior': []
        }

        # 标识恶意用户 (0=正常, 1=恶意)
        self.malicious_users = np.zeros(num_users, dtype=int)
        malicious_indices = np.random.choice(num_users, self.num_malicious, replace=False)
        self.malicious_users[malicious_indices] = 1

    def _generate_messages(self):
        """生成模拟消息和签名"""
        messages = []
        signatures = []

        for user_id in range(self.num_users):
            # 生成随机消息 (模拟模型更新)
            message = f"Update_{user_id}_{np.random.rand()}".encode('utf-8')

            if self.malicious_users[user_id] == 1:
                # 恶意用户行为模式
                behavior_type = np.random.choice(['tamper', 'invalid_signature', 'replay'])

                if behavior_type == 'tamper':
                    pass
                    # # 篡改消息内容
                    # message = f"Malicious_{user_id}_{np.random.rand()}".encode('utf-8')
                    # signature = self.pki.sign(user_id, message)
                elif behavior_type == 'invalid_signature':
                    # 使用无效签名
                    signature = b'invalid_signature' + str(np.random.rand()).encode()
                elif behavior_type == 'replay':
                    # 重放攻击：重复使用旧签名
                    if hasattr(self, 'old_signatures') and user_id in self.old_signatures:
                        signature = self.old_signatures[user_id]
                    else:
                        signature = self.pki.sign(user_id, message)
            else:
                # 正常用户行为
                signature = self.pki.sign(user_id, message)

            messages.append(message)
            signatures.append(signature)

        # 保存当前签名用于重放攻击
        self.old_signatures = signatures.copy()

        return messages, signatures

    def run(self):
        """运行实验"""
        for round_idx in range(self.num_rounds):
            # 1. 生成模拟数据
            snrs = np.random.normal(20, 5, self.num_users)
            delta_omegas = np.abs(np.random.normal(1.0, 0.3, self.num_users))
            messages, signatures = self._generate_messages()

            # 2. 执行安全验证
            valid_mask = self.selector.verify_updates(messages, signatures)

            # 3. 记录验证结果
            true_labels = self.malicious_users.copy()
            pred_labels = 1 - valid_mask.astype(int)  # 0=有效, 1=无效(恶意)

            # 计算指标
            tp = np.sum((pred_labels == 1) & (true_labels == 1))
            tn = np.sum((pred_labels == 0) & (true_labels == 0))
            fp = np.sum((pred_labels == 1) & (true_labels == 0))
            fn = np.sum((pred_labels == 0) & (true_labels == 1))

            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

            self.results['detection_accuracy'].append(accuracy)
            self.results['false_positive_rate'].append(fpr)
            self.results['false_negative_rate'].append(fnr)

            # 4. 计算信任评分差异
            trust_scores = np.array([self.selector._calculate_trust_score(u) for u in range(self.num_users)])
            normal_trust = np.mean(trust_scores[self.malicious_users == 0])
            malicious_trust = np.mean(trust_scores[self.malicious_users == 1])
            trust_diff = normal_trust - malicious_trust

            self.results['trust_score_diff'].append(trust_diff)
            self.results['malicious_behavior'].append(np.sum(pred_labels == 1))

            # 5. 执行异常检测
            anomalies = self.selector.detect_anomalies(threshold=3.0)
            self.results['anomalies_detected'] = len(anomalies)

            # 打印轮次结果
            print(f"Round {round_idx + 1}/{self.num_rounds}: "
                  f"Acc={accuracy:.2f}, FPR={fpr:.2f}, FNR={fnr:.2f}, "
                  f"TrustDiff={trust_diff:.2f}, MaliciousDetected={np.sum(pred_labels == 1)}")

    def plot_results(self):
        """可视化实验结果 - Times New Roman字体，PDF输出"""
        # 设置Times New Roman字体
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 20
        plt.rcParams['axes.titlesize'] = 22
        plt.rcParams['axes.labelsize'] = 20

        plt.figure(figsize=(36, 24))  # 调整尺寸以适应PDF
        # 调整整体布局
        plt.subplots_adjust(left=0.07, right=0.93, bottom=0.1, top=0.95, hspace=0.35, wspace=0.15)

        # 1. 检测准确率
        ax1 = plt.subplot(2, 2, 1)
        plt.plot(self.results['detection_accuracy'], 'b-o', lw=2, ms=10, label='Accuracy')
        plt.plot(self.results['false_positive_rate'], 'r--x', lw=2, ms=10, label='False Positive Rate')
        plt.plot(self.results['false_negative_rate'], 'g-.s', lw=2, ms=8, label='False Negative Rate')
        ax1.set_xlabel('Round', fontsize=20)
        ax1.set_ylabel('Rate', fontsize=20)
        ax1.legend(prop={'family': 'Times New Roman'})
        ax1.grid(True)
        ax1.set_ylim(-0.05, 1.05)
        ax1.set_title('(a) Signature Verification Performance', y=-0.15, fontsize=22, fontweight='bold')

        # 2. 信任评分差异
        ax2 = plt.subplot(2, 2, 2)
        plt.plot(self.results['trust_score_diff'], 'm-^', lw=2, ms=10)
        ax2.set_xlabel('Round', fontsize=20)
        ax2.set_ylabel('Trust Score Difference', fontsize=20)
        ax2.grid(True)
        ax2.set_title('(b) Trust Score Difference (Normal vs Malicious)', y=-0.15, fontsize=22, fontweight='bold')

        # 3. 恶意行为检测
        ax3 = plt.subplot(2, 2, 3)
        malicious_count = self.num_malicious * np.ones(self.num_rounds)
        detected_malicious = np.array(self.results['malicious_behavior'])
        plt.plot(detected_malicious, 'r-o', lw=2, ms=8, label='Detected Malicious')
        plt.plot(malicious_count, 'b--', lw=3, label='Actual Malicious')
        ax3.set_xlabel('Round', fontsize=20)
        ax3.set_ylabel('Count', fontsize=20)
        ax3.legend(prop={'family': 'Times New Roman'})
        ax3.grid(True)
        ax3.set_title('(c) Malicious Behavior Detection', y=-0.15, fontsize=22, fontweight='bold')

        # 4. 行为画像示例
        ax4 = plt.subplot(2, 2, 4)
        normal_user = np.random.choice(np.where(self.malicious_users == 0)[0])
        malicious_user = np.random.choice(np.where(self.malicious_users == 1)[0])

        normal_profile = self.selector.behavior_profiles[normal_user]
        malicious_profile = self.selector.behavior_profiles[malicious_user]

        # 归一化处理
        normal_profile_normalized = normal_profile / np.sum(normal_profile)
        malicious_profile_normalized = malicious_profile / np.sum(malicious_profile)

        bar_width = 0.25
        x = np.arange(3)
        rects1 = ax4.bar(x - bar_width / 2, normal_profile_normalized,
                         width=bar_width, label=f'Normal User {normal_user}',
                         color='skyblue', edgecolor='black')
        rects2 = ax4.bar(x + bar_width / 2, malicious_profile_normalized,
                         width=bar_width, label=f'Malicious User {malicious_user}',
                         color='salmon', edgecolor='black')

        ax4.set_xticks(x)
        ax4.set_xticklabels(['Sign Count', 'Verify Success', 'Verify Fail'], fontsize=18)
        ax4.set_ylabel('Proportion', fontsize=20)
        ax4.legend(prop={'family': 'Times New Roman'})
        ax4.grid(True, axis='y')
        max_val = max(max(normal_profile_normalized), max(malicious_profile_normalized))
        ax4.set_ylim(0, max_val * 1.25)

        # 添加柱状图数值标签
        for rect in rects1:
            height = rect.get_height()
            ax4.text(rect.get_x() + rect.get_width() / 2, height + 0.02,
                     f'{height:.2f}', ha='center', va='bottom',
                     fontsize=16, family='Times New Roman', fontweight='bold')

        for rect in rects2:
            height = rect.get_height()
            ax4.text(rect.get_x() + rect.get_width() / 2, height + 0.02,
                     f'{height:.2f}', ha='center', va='bottom',
                     fontsize=16, family='Times New Roman', fontweight='bold')

        ax4.set_title('(d) Behavior Profiles (Normalized)', y=-0.15, fontsize=22, fontweight='bold')

        # 保存为PDF矢量图
        plt.savefig('signature_verification_results.pdf', dpi=300, bbox_inches='tight')
        plt.show()

    def run_detailed_analysis(self):
        """运行详细分析"""
        # 测试1: 密钥轮换验证
        print("\nTesting Key Rotation...")
        user_id = 0
        orig_public_key = self.pki.key_pairs[user_id]['public']

        # 模拟大量签名触发轮换
        for i in range(1500):
            message = f"Test_{i}".encode()
            self.pki.sign(user_id, message)

        new_public_key = self.pki.key_pairs[user_id]['public']
        print(f"Key changed: {orig_public_key != new_public_key}")

        # 测试2: 重放攻击检测
        print("\nTesting Replay Attack...")
        message = b"Legitimate message"
        valid_signature = self.pki.sign(user_id, message)

        # 尝试重放旧签名
        replay_result = self.pki.verify(user_id, message, valid_signature)
        print(f"Initial verification: {replay_result}")

        # 更新密钥后重放
        for i in range(1500):
            self.pki.sign(user_id, f"Rotation_{i}".encode())

        replay_result = self.pki.verify(user_id, message, valid_signature)
        print(f"Replay attack after key rotation: {replay_result} (should be False)")

        # 测试3: 异常检测效果
        print("\nTesting Anomaly Detection...")
        anomalies = self.selector.detect_anomalies(threshold=3.0)
        detected_malicious = [u for u in anomalies if self.malicious_users[u] == 1]
        false_positives = [u for u in anomalies if self.malicious_users[u] == 0]

        print(f"Detected {len(anomalies)} anomalies: "
              f"{len(detected_malicious)} malicious, {len(false_positives)} false positives")

        # 计算异常检测指标
        if anomalies:
            precision = len(detected_malicious) / len(anomalies)
            recall = len(detected_malicious) / self.num_malicious
            print(f"Anomaly detection precision: {precision:.2f}, recall: {recall:.2f}")


if __name__ == "__main__":
    print("Starting Signature Verification Experiment...")
    experiment = SignatureVerificationExperiment(
        num_users=100,
        malicious_ratio=0.05,
        num_rounds=100
    )

    # 运行主实验
    experiment.run()

    # 运行详细分析
    experiment.run_detailed_analysis()

    # 可视化结果
    experiment.plot_results()
    print("Experiment completed. Results saved to signature_verification_results.png")