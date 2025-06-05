#experiment.py
import time
from tqdm import tqdm
from data_utils import *
from channel_simulator import *
from hsfl import *
from ue_selection import *


class ExperimentRunner:
    def __init__(self, config):
        # 实验配置
        self.config = {
            'num_users': 100,
            'rounds': 100,
            'model_type': 'Net',
            'split_ratio': 0.5,
            'selection_scheme': 'MAB-BC-BN2',
            'data_dist': 'noniid',
            'batch_size': 32,
            'local_epochs': 5,
            'seed': 42
        }
        self.config.update(config)
        self._init_seed()

        # 初始化组件
        self._prepare_data()
        self._init_channel()
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
        self.train_set, self.test_set = load_mnist()

        if self.config['data_dist'] == 'iid':
            self.train_set.partitions = iid_partition(self.train_set,
                                                      self.config['num_users'])
        elif self.config['data_dist'] == 'noniid':
            self.train_set.partitions = noniid_partition(self.train_set,
                                                         self.config['num_users'])

    def _init_channel(self):
        self.channel = ChannelSimulator(environment='urban')
        self.ue_positions = self.channel.generate_ue_positions(
            self.config['num_users'], static=False)

    def _init_trainers(self):
        common_args = {
            'num_users': self.config['num_users'],
            'train_set': self.train_set,
            'test_set': self.test_set,
            'model_type': self.config['model_type'],
            'batch_size': self.config['batch_size'],
            'local_epochs': self.config['local_epochs']
        }

        self.hsfl = HSFLTrainer(**common_args,
                                split_ratio=self.config['split_ratio'])

    def _init_selector(self):
        self.selector = MABSelector(
            num_users=self.config['num_users'],
            beta=0.5,
            lambda_l=0.9,
            lambda_c=0.9
        )

    # def _calculate_communication(self, split_users):
    #     split_users = np.array(split_users)
    #     # 计算HSFL通信开销
    #     model_size = sum(p.numel() for p in self.hsfl.global_model.parameters())
    #     split_layer_size = np.prod(self.hsfl.split_layer_shape)
    #
    #     # 联邦训练用户传输完整模型
    #     fed_comm = (model_size * 32) / 1e6  # MB
    #     # 分割训练用户传输激活值+梯度
    #     split_comm = (split_layer_size * 32 * 2) / 1e6  # MB
    #
    #     return fed_comm * (len(split_users) - split_users.size) + \
    #         split_comm * split_users.size

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
        for round in tqdm(range(self.config['rounds'])):
            # 更新UE位置并获取信道质量
            self.channel.update_positions()
            snrs = self.channel.calculate_snr(self.ue_positions)

            # 用户选择
            if self.config['selection_scheme'] == 'random':
                selected = np.random.choice(
                    self.config['num_users'],
                    size=self.config['k'],
                    replace=False
                )
                split_users = np.random.choice(
                    selected,
                    size=int(self.config['k'] * self.config['split_ratio']),
                    replace=False
                )
            else:
                # 需要先获取模型更新量（这里简化为随机数）
                delta_norms = np.random.rand(self.config['num_users'])
                self.selector.update(range(self.config['num_users']), delta_norms, snrs)
                selected = self.selector.select_users(self.config['k'])
                split_users = selected[:int(self.config['k'] * self.config['split_ratio'])]

            # 运行HSFL
            start_time = time.time()
            hsfl_acc = self.hsfl.train_round(selected, split_users)
            hsfl_time = time.time() - start_time


            # 记录结果
            self.results['hsfl']['acc'].append(hsfl_acc)
            self.results['hsfl']['time'].append(hsfl_time)
            self.results['hsfl']['comm'].append(
                self._calculate_communication(split_users))

        return self.results


if __name__ == "__main__":
    config0 = {
        'rounds': 100,
        'k': 10,
        'data_dist': 'noniid',
        'selection_scheme': 'MAB-BC-BN2',
        'model_type': 'Net'
    }

    runner = ExperimentRunner(config0)
    results = runner.run()
    torch.save(results, 'experiment_results0.pt')

    config1 = {
        'rounds': 100,
        'k': 10,
        'data_dist': 'noniid',
        'selection_scheme': 'MAB-BC-BN2',
        'model_type': 'AlexNet'
    }

    runner = ExperimentRunner(config1)
    results = runner.run()
    torch.save(results, 'experiment_results1.pt')

    config2 = {
        'rounds': 100,
        'k': 10,
        'data_dist': 'iid',
        'selection_scheme': 'MAB-BC-BN2',
        'model_type': 'Net'
    }

    runner = ExperimentRunner(config2)
    results = runner.run()
    torch.save(results, 'experiment_results2.pt')

    config3 = {
        'rounds': 100,
        'k': 10,
        'data_dist': 'iid',
        'selection_scheme': 'MAB-BC-BN2',
        'model_type': 'AlexNet'
    }

    runner = ExperimentRunner(config3)
    results = runner.run()
    torch.save(results, 'experiment_results3.pt')

