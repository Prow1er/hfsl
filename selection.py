# selection.py
import numpy as np
import torch
from ue_selection_new import calculate_delta_norms
from secure_selector import PKIManager, SecureMABSelector


def simulate_fl_round(selector, pki, num_users=100, K=10):
    """
    模拟联邦学习的一轮训练，返回选中的用户、SNR、模型差异及异常列表
    """
    # 模拟全局模型
    global_model = {'linear.weight': torch.randn(10, 10), 'linear.bias': torch.randn(10)}

    # 客户端生成带签名的更新
    messages, signatures, local_models = [], [], []
    for u in range(num_users):
        # 模拟本地模型更新
        local_model = {
            'linear.weight': global_model['linear.weight'] + torch.randn(10, 10) * 0.1,
            'linear.bias': global_model['linear.bias'] + torch.randn(10) * 0.1
        }
        local_models.append(local_model)

        # 生成签名
        msg = str(local_model).encode()
        try:
            sig = pki.sign(u, msg)
        except ValueError:  # 处理吊销用户
            sig = b''
        messages.append(msg)
        signatures.append(sig)

    # 服务器处理
    delta_norms = calculate_delta_norms(global_model, local_models)
    snrs = np.random.normal(20, 5, num_users)

    # 选择用户
    selected = selector.select_users(
        K=K,
        snrs=snrs,
        delta_omegas=delta_norms,
        messages=messages,
        signatures=signatures
    )

    # 更新选择器状态
    valid = selector.verify_updates(messages, signatures)
    selector.update(selected, delta_norms * valid, snrs * valid)

    # 检测异常
    anomalies = selector.detect_anomalies()
    return selected, snrs, delta_norms, anomalies


if __name__ == "__main__":
    NUM_USERS = 100
    pki = PKIManager(NUM_USERS)
    selector = SecureMABSelector(
        pki_manager=pki,
        num_users=NUM_USERS,
        alpha=0.1,
        beta=0.5,
        T_alliance=5
    )

    # 模拟运行10轮
    for round in range(20):
        # 执行一轮联邦学习并获取数据
        selected, snrs, delta_norms, anomalies = simulate_fl_round(selector, pki)

        # 按照ue_selection_new.py的格式输出
        print(f"Round {round + 1} selected users:")
        print("Indices:", selected)
        print("SNRs:", np.round(snrs[selected], 1))
        print("Norms:", np.round(delta_norms[selected], 2))
        print("Current counts:", np.round(selector.counts[selected], 2))

        # 处理异常用户吊销
        # print(f"检测到异常用户: {anomalies}")
        if round % 3 == 2 and len(anomalies) > 0:
            revoked = anomalies[0]
            pki.revoked.add(revoked)
            print(f"Revoked user {revoked}")
        print("---")