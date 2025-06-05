# experiment_new.py
import time
import numpy as np
import torch
from tqdm import tqdm
from data_utils import load_mnist, iid_partition, noniid_partition
from channel_simulator import ChannelSimulator
from hsfl_new import HSFLTrainer
from secure_selector import PKIManager, SecureMABSelector
from ue_selection_new import MABSelector, calculate_delta_norms


class ExperimentRunner:
    def __init__(self, config):
        # 实验配置
        self.config = {
            'num_users': 100,
            'rounds': 100,
            'model_type': 'Net',
            'split_ratio': 0.5,
            'selection_scheme': 'secure',
            'data_dist': 'noniid',
            'batch_size': 32,
            'local_epochs': 5,
            'k': 10,
            'seed': 42
        }
        self.config.update(config)
        self._init_seed()

        # 初始化组件
        self._prepare_data()
        self._init_channel()
        self._init_pki()
        self._init_trainers()
        self._init_selector()

        # 结果记录
        self.results = {
            'hsfl': {'acc': [], 'time': [], 'comm': []},
        }

    def _init_seed(self):
        np.random.seed(self.config['seed'])
        torch.manual_seed(self.config['seed'])

    def _prepare_data(self):
        """加载并分区数据集"""
        self.train_set, self.test_set = load_mnist()

        if self.config['data_dist'] == 'iid':
            partitions = iid_partition(self.train_set, self.config['num_users'])
        else:
            partitions = noniid_partition(self.train_set, self.config['num_users'])

        self.train_set.partitions = partitions  # 适配新的PartitionedDataset

    def _init_channel(self):
        """初始化无线信道模拟器"""
        self.channel = ChannelSimulator(environment='urban')
        self.ue_positions = self.channel.generate_ue_positions(
            self.config['num_users'],
            static=self.config.get('static_ue', False)
        )

    def _init_pki(self):
        """初始化公钥基础设施"""
        self.pki_manager = PKIManager(self.config['num_users'])

    def _init_trainers(self):
        """初始化HSFL训练器"""
        self.hsfl = HSFLTrainer(
            num_users=self.config['num_users'],
            train_set=self.train_set,
            test_set=self.test_set,
            model_type=self.config['model_type'],
            split_ratio=self.config['split_ratio'],
            batch_size=self.config['batch_size'],
            local_epochs=self.config['local_epochs']
        )

    def _init_selector(self):
        """初始化用户选择器"""
        if 'secure' in self.config['selection_scheme']:
            self.selector = SecureMABSelector(
                pki_manager=self.pki_manager,
                num_users=self.config['num_users'],
                beta=0.5,
                lambda_l=0.9,
                lambda_c=0.9
            )
        else:
            self.selector = MABSelector(
                num_users=self.config['num_users'],
                beta=0.5,
                lambda_l=0.9,
                lambda_c=0.9
            )

    def _get_client_updates(self):
        """模拟获取客户端更新（用于签名验证）"""
        messages = [b'update_' + str(i).encode() for i in range(self.config['num_users'])]
        signatures = [self.pki_manager.sign(i, msg) for i, msg in enumerate(messages)]
        return messages, signatures

    def _calculate_communication(self, split_users):
        """计算通信开销"""
        split_layer_size = np.prod(self.hsfl.split_layer_shape)
        fed_comm = (sum(p.numel() for p in self.hsfl.global_model.parameters()) * 32) / 1e6  # MB
        split_comm = (split_layer_size * 32 * 2) / 1e6  # MB

        num_split = len(split_users)
        num_total = self.config['k']
        num_fed = num_total - num_split

        return num_fed * fed_comm + num_split * split_comm

    def run(self):
        """执行完整实验流程"""
        for round in tqdm(range(self.config['rounds']), desc="Training Rounds"):
            # 1. 信道状态更新
            self.channel.update_positions()
            snrs = self.channel.calculate_snr(self.ue_positions)

            # 2. 获取客户端更新签名
            messages, signatures = self._get_client_updates()

            # 3. 用户选择
            if isinstance(self.selector, SecureMABSelector):
                # 安全选择需要验证签名
                global_state = self.hsfl.get_global_model_state()
                local_states = [global_state] * self.config['num_users']  # 模拟本地更新
                delta_norms = calculate_delta_norms(global_state, local_states)

                selected = self.selector.select_users(
                    K=self.config['k'],
                    snrs=snrs,
                    delta_omegas=delta_norms,
                    messages=messages,
                    signatures=signatures
                )
            else:
                # 常规MAB选择
                delta_norms = np.random.rand(self.config['num_users'])
                self.selector.update(range(self.config['num_users']), delta_norms, snrs)
                selected = self.selector.select_users(self.config['k'])

            # 4. 确定分割训练用户
            split_users = np.random.choice(
                selected,
                size=int(len(selected) * self.config['split_ratio']),
                replace=False
            )

            # 5. HSFL训练
            start_time = time.time()
            acc = self.hsfl.train_round(selected, split_users)
            round_time = time.time() - start_time

            # 6. 记录结果
            self.results['hsfl']['acc'].append(acc)
            self.results['hsfl']['time'].append(round_time)
            self.results['hsfl']['comm'].append(
                self._calculate_communication(split_users)
            )

            # 7. 异常检测
            if isinstance(self.selector, SecureMABSelector):
                anomalies = self.selector.detect_anomalies()
                if len(anomalies) > 0:
                    print(f"Round {round + 1} detected anomalies: {anomalies}")

        return self.results


if __name__ == "__main__":
    config0 = {
        'rounds': 100,
        'k': 10,
        'model_type': 'Net',
        'selection_scheme': 'secure',
        'data_dist': 'noniid',
        'static_ue': False  # 是否使用静态用户设备位置
    }

    runner = ExperimentRunner(config0)
    results = runner.run()
    torch.save(results, 'secure_hsfl_results0.pt')

    config1 = {
        'rounds': 100,
        'k': 10,
        'model_type': 'AlexNet',
        'selection_scheme': 'secure',
        'data_dist': 'noniid',
        'static_ue': False  # 是否使用静态用户设备位置
    }

    runner = ExperimentRunner(config1)
    results = runner.run()
    torch.save(results, 'secure_hsfl_results1.pt')

    config2 = {
        'rounds': 100,
        'k': 10,
        'model_type': 'Net',
        'selection_scheme': 'secure',
        'data_dist': 'iid',
        'static_ue': False  # 是否使用静态用户设备位置
    }

    runner = ExperimentRunner(config2)
    results = runner.run()
    torch.save(results, 'secure_hsfl_results2.pt')

    config3 = {
        'rounds': 100,
        'k': 10,
        'model_type': 'AlexNet',
        'selection_scheme': 'secure',
        'data_dist': 'iid',
        'static_ue': False  # 是否使用静态用户设备位置
    }

    runner = ExperimentRunner(config3)
    results = runner.run()
    torch.save(results, 'secure_hsfl_results3.pt')