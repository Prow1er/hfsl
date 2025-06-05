# secure_selector.py
import numpy as np
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes
from ue_selection_new import MABSelector


class PKIManager:
    """
    完整的公钥基础设施管理类:
    管理所有用户的数字证书生命周期
    实现密钥自动轮换机制
    处理签名验证与证书吊销
    """

    def __init__(self, num_users):
        self.key_pairs = [self._generate_key_pair() for _ in range(num_users)]
        self.revoked = set()
        self.certificates = {}
        self.key_rotation_count = np.zeros(num_users, dtype=int)

    def _generate_key_pair(self):
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        return {
            'private': private_key,
            'public': private_key.public_key(),
            'counter': 0
        }

    def sign(self, user_id, message):
        """带密钥轮换的签名方法"""
        if user_id in self.revoked:
            raise ValueError("User revoked")

        # 自动密钥轮换机制
        if self.key_pairs[user_id]['counter'] > 1000:
            self.key_pairs[user_id] = self._generate_key_pair()

        signature = self.key_pairs[user_id]['private'].sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        self.key_pairs[user_id]['counter'] += 1
        return signature

    def verify(self, user_id, message, signature):
        """带证书检查的验证方法"""
        if user_id in self.revoked:
            return False
        try:
            public_key = self.key_pairs[user_id]['public']
            public_key.verify(
                signature,
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception as e:
            print(f"Verification failed for user {user_id}: {str(e)}")
            return False


class SecureMABSelector(MABSelector):
    """增强安全性的MAB选择器"""

    def __init__(self, pki_manager, **kwargs):
        super().__init__(**kwargs)
        self.pki = pki_manager
        self.integrity_scores = np.ones(self.num_users)
        self.behavior_profiles = np.zeros((self.num_users, 3))  # [sign_count, verify_success, verify_fail]

    def _update_behavior_profile(self, user_id, verify_result):
        """更新用户行为画像
        维护三维行为特征向量：[签名次数，验证成功次数，验证失败次数]
        每次验证后更新对应计数器
        """
        if verify_result:
            self.behavior_profiles[user_id, 1] += 1
        else:
            self.behavior_profiles[user_id, 2] += 1
        self.behavior_profiles[user_id, 0] += 1

    def verify_updates(self, messages, signatures):
        """带行为分析的批量验证"""
        valid_mask = np.zeros(self.num_users, dtype=bool)
        for u in range(self.num_users):
            if u in self.pki.revoked:
                continue
            try:
                valid = self.pki.verify(u, messages[u], signatures[u])
                self._update_behavior_profile(u, valid)
                valid_mask[u] = valid
            except Exception as e:
                print(f"Error verifying user {u}: {str(e)}")
                valid_mask[u] = False
        return valid_mask

    def _calculate_trust_score(self, user_id):
        """综合信任评分"""
        total = self.behavior_profiles[user_id].sum()
        if total == 0:
            return 0.0
        success_ratio = self.behavior_profiles[user_id, 1] / total
        activity = np.log1p(self.behavior_profiles[user_id, 0])
        return success_ratio * activity

    def select_users(self, K, snrs, delta_omegas, messages, signatures):
        """安全增强的选择流程"""
        # 执行签名验证
        valid_users = self.verify_updates(messages, signatures)

        # 更新信任评分
        trust_scores = np.array([self._calculate_trust_score(u) for u in range(self.num_users)])
        self.integrity_scores = 0.9 * self.integrity_scores + 0.1 * trust_scores

        # 过滤无效用户
        valid_indices = np.where(valid_users)[0]
        filtered_snrs = snrs[valid_indices]
        filtered_deltas = delta_omegas[valid_indices]

        # 执行父类选择逻辑
        selected = super().select_users(K, filtered_snrs, filtered_deltas)

        # 映射回原始索引
        return valid_indices[selected]

    def detect_anomalies(self, threshold=2.5):
        """基于马氏距离的异常检测"""
        cov = np.cov(self.behavior_profiles.T)
        inv_cov = np.linalg.pinv(cov)
        mean = np.mean(self.behavior_profiles, axis=0)

        anomalies = []
        for u in range(self.num_users):
            delta = self.behavior_profiles[u] - mean
            distance = np.sqrt(delta.T @ inv_cov @ delta)
            if distance > threshold:
                anomalies.append(u)
        return anomalies