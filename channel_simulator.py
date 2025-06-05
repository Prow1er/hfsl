#channel_simulator.py
import numpy as np


class ChannelSimulator:
    def __init__(self,
                 cell_radius=500,
                 bs_height=20,
                 ue_height_range=(20, 80),
                 environment='urban',
                 carrier_freq=2.4e9):
        """
        初始化无线信道模拟器

        参数：
            cell_radius: 小区半径（米）
            bs_height: 基站天线高度（米）
            ue_height_range: (UE最小高度, UE最大高度)（米）
            environment: 环境类型 ['rural', 'suburban', 'urban', 'dense_urban']
            carrier_freq: 载波频率（Hz）
        """
        self.cell_radius = cell_radius
        self.bs_height = bs_height
        self.ue_height_range = ue_height_range
        self.carrier_freq = carrier_freq
        self.light_speed = 3e8  # 光速（m/s）

        # 根据环境类型设置参数（参考论文[20][21]）
        self.env_params = {
            'rural': {'a': 4.88, 'b': 0.43, 'alpha_los': 2.1, 'alpha_nlos': 3.0,
                      'phi_l': 1.0, 'phi_n': 20.0},
            'suburban': {'a': 9.65, 'b': 0.16, 'alpha_los': 2.5, 'alpha_nlos': 3.5,
                         'phi_l': 1.6, 'phi_n': 23.0},
            'urban': {'a': 12.08, 'b': 0.11, 'alpha_los': 2.8, 'alpha_nlos': 4.0,
                      'phi_l': 2.3, 'phi_n': 34.0},
            'dense_urban': {'a': 27.23, 'b': 0.08, 'alpha_los': 3.5, 'alpha_nlos': 4.5,
                            'phi_l': 3.0, 'phi_n': 40.0}
        }
        self.set_environment(environment)

        # 预计算波长
        self.wavelength = self.light_speed / self.carrier_freq

    def set_environment(self, environment):
        """更新环境参数"""
        assert environment in self.env_params, f"Invalid environment: {environment}"
        params = self.env_params[environment]
        self.a = params['a']
        self.b = params['b']
        self.alpha_los = params['alpha_los']
        self.alpha_nlos = params['alpha_nlos']
        self.phi_l = params['phi_l']
        self.phi_n = params['phi_n']
        self.current_env = environment

    def generate_ue_positions(self, num_ues, static=False):
        """
        生成UE初始位置
        参数：
            num_ues: UE数量
            static: 是否生成静态位置（否则按论文设置随时间变化）
        返回：
            ue_positions: (N, 3)数组，每行包含(x, y, h)
        """
        # 水平位置均匀分布在圆形区域内
        angles = np.random.uniform(0, 2 * np.pi, num_ues)
        radii = np.sqrt(np.random.uniform(0, 1, num_ues)) * self.cell_radius
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)

        # 高度均匀分布
        h = np.random.uniform(*self.ue_height_range, num_ues)

        self.ue_positions = np.column_stack((x, y, h))
        self.velocity = None if static else self._init_mobility()
        return self.ue_positions.copy()

    def _init_mobility(self, speed_range=(5, 15)):
        """初始化移动速度（m/s）"""
        num_ues = len(self.ue_positions)
        speeds = np.random.uniform(*speed_range, num_ues)
        angles = np.random.uniform(0, 2 * np.pi, num_ues)
        return np.column_stack((speeds * np.cos(angles),
                                speeds * np.sin(angles),
                                np.zeros(num_ues)))  # 假设水平移动

    def update_positions(self, delta_t=1.0):
        """更新UE位置（如果移动）"""
        if self.velocity is not None:
            self.ue_positions += self.velocity * delta_t
            # 保持高度在范围内
            self.ue_positions[:, 2] = np.clip(self.ue_positions[:, 2],
                                              *self.ue_height_range)

    def calculate_elevation_angle(self, ue_positions):
        """
        计算仰角（度）
        参数：
            ue_positions: (N,3)数组，UE坐标(x,y,h)
        返回：
            elevation_angles: (N,)数组，仰角（度）
        """
        dx = np.linalg.norm(ue_positions[:, :2], axis=1)  # 水平距离
        dh = ue_positions[:, 2] - self.bs_height  # 高度差
        distance_3d = np.sqrt(dx ** 2 + dh ** 2)  # 3D距离
        return np.degrees(np.arcsin(dh / distance_3d))

    def los_probability(self, ue_positions):
        """
        计算LoS概率（公式1）
        返回：
            prob_los: (N,)数组，每个UE的LoS概率
        """
        theta = self.calculate_elevation_angle(ue_positions)
        return 1 / (1 + self.a * np.exp(-self.b * (theta - self.a)))

    def path_loss(self, ue_positions):
        """
        计算路径损耗（dB）（公式2）
        返回：
            path_loss: (N,)数组，路径损耗值（dB）
        """
        # 计算3D距离
        dx = np.linalg.norm(ue_positions[:, :2], axis=1)
        dh = ue_positions[:, 2] - self.bs_height
        distance_3d = np.sqrt(dx ** 2 + dh ** 2)

        # 自由空间路径损耗基础值
        fsp1 = 20 * np.log10(4 * np.pi * distance_3d / self.wavelength)

        # LoS和NLoS附加损耗
        pl_los = fsp1 + self.phi_l + 10 * self.alpha_los * np.log10(distance_3d)
        pl_nlos = fsp1 + self.phi_n + 10 * self.alpha_nlos * np.log10(distance_3d)

        # 计算加权平均
        prob_los = self.los_probability(ue_positions)
        return prob_los * pl_los + (1 - prob_los) * pl_nlos

    def calculate_snr(self, ue_positions, tx_power=20, noise_figure=7):
        """
        计算接收SNR（dB）
        参数：
            tx_power: UE发射功率（dBm）
            noise_figure: 接收机噪声系数（dB）
        返回：
            snr: (N,)数组，SNR（dB）
        """
        # 转换单位为dBm
        tx_power_dbm = tx_power
        noise_dbm = -174 + 10 * np.log10(20e6) + noise_figure  # 假设20MHz带宽

        # 计算接收功率
        path_loss = self.path_loss(ue_positions)
        rx_power = tx_power_dbm - path_loss

        # 计算SNR（考虑瑞利衰落）
        snr = rx_power - noise_dbm
        # 添加小尺度衰落（瑞利分布，假设平均功率为0 dB）
        snr += 10 * np.log10(np.random.rayleigh(size=len(ue_positions)))
        return snr


if __name__ == "__main__":
    # 示例用法
    np.random.seed(42)

    # 初始化信道模拟器（城市环境）
    chan = ChannelSimulator(environment='urban')

    # 生成10个