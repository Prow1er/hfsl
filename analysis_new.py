import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.gridspec as gridspec
import matplotlib as mpl  # 导入matplotlib


class ResultAnalyzer:
    def __init__(self, result_paths, labels=None):
        """
        初始化结果分析器

        Args:
            result_paths (list): 结果文件路径列表
            labels (list): 每个结果的标签，默认为文件名
        """
        if not isinstance(result_paths, list):
            result_paths = [result_paths]

        self.results = []
        for path in result_paths:
            self.results.append(torch.load(path))

        if labels is None:
            self.labels = [os.path.splitext(os.path.basename(path))[0] for path in result_paths]
        else:
            self.labels = labels

        # 全局设置Times New Roman字体
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 28  # 比之前稍大以适应Times New Roman

        # 使用蓝色和橙色的专业调色板替代棕色
        self.colors = ['#1f77b4', '#ff7f0e'][:len(result_paths)]

        # 确保Seaborn不会覆盖字体设置
        sns.set_style("whitegrid", {'font.family': 'Times New Roman'})

    def plot_accuracy_curves(self, algorithms=['hsfl']):
        """绘制优化后的准确率曲线和时间柱状图（横向）"""
        # 再次确保Times New Roman设置
        plt.rcParams['font.family'] = 'Times New Roman'

        # 创建带有特定高度比例的子图 (65% vs 35%)
        plt.figure(figsize=(32, 22))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

        # 增加子图间距
        plt.subplots_adjust(left=0.08, right=0.92, bottom=0.08, top=0.92, hspace=0.35)

        # 第一个子图：准确率曲线 (65%)
        ax1 = plt.subplot(gs[0])

        for idx, (result, label) in enumerate(zip(self.results, self.labels)):
            for algo in algorithms:
                if algo in result:
                    acc = np.array(result[algo]['acc'])
                    x = np.arange(len(acc))

                    # 添加平滑处理
                    window_size = 5
                    if len(acc) > window_size:
                        acc_smooth = np.convolve(acc, np.ones(window_size) / window_size, mode='valid')
                        line, = plt.plot(x[window_size - 1:], acc_smooth,
                                         label=f'{label}',
                                         color=self.colors[idx],
                                         linewidth=6,
                                         marker='o' if idx == 0 else 's',
                                         markersize=20,  # 从8增加到14
                                         markevery=10)

                        # 标记关键点
                        # 标记最高点
                        max_idx = np.argmax(acc_smooth)
                        plt.scatter(x[window_size - 1:][max_idx], acc_smooth[max_idx],
                                    s=200, color=self.colors[idx], zorder=10,
                                    edgecolor='black')
                        plt.annotate(f'{acc_smooth[max_idx]:.4f}',
                                     (x[window_size - 1:][max_idx], acc_smooth[max_idx]),
                                     textcoords="offset points", xytext=(0, 15),
                                     ha='center', fontsize=26, weight='bold',
                                     arrowprops=dict(arrowstyle="->", color='black'),
                                     family='Times New Roman')  # 添加字体设置

                        # 添加渐变填充
                        plt.fill_between(x[window_size - 1:], 0, acc_smooth,
                                         color=self.colors[idx], alpha=0.15)

        # 美学优化
        ax1.set_facecolor('#f8f8f8')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_xlabel("Communication Rounds", fontsize=34, labelpad=15, family='Times New Roman')
        ax1.set_ylabel("Test Accuracy", fontsize=34, labelpad=15, family='Times New Roman')
        ax1.set_title("Learning Performance Comparison", pad=20, fontsize=38, family='Times New Roman',
                      fontweight='bold')

        # 设置图例为Times New Roman
        ax1.legend(loc='lower right', fontsize=30, framealpha=0.9, shadow=True, prop={'family': 'Times New Roman'})

        # 第二个子图：总时间柱状图 (35%，横向显示)
        ax2 = plt.subplot(gs[1])

        # 计算每个实验的总时间
        total_times = []
        for result in self.results:
            for algo in algorithms:
                if algo in result:
                    total_time = np.sum(result[algo]['time'])
                    total_times.append(total_time)

        # 计算时间差和百分比
        if len(total_times) == 2:
            time_diff = total_times[1] - total_times[0]
            diff_percent = abs(time_diff / total_times[0]) * 100
            diff_sign = "+" if time_diff > 0 else "-"

        # 创建横向柱状图 - 更高级的样式
        y_pos = np.arange(len(total_times))
        bars = ax2.barh(y_pos, total_times, color=self.colors[:len(total_times)],
                        height=0.5, edgecolor='black', linewidth=3)

        # 添加渐变效果
        for i, bar in enumerate(bars):
            bar.set_hatch('////')
            if i == 1:
                bar.set_hatch('\\\\\\\\')

        # 添加数值标签 - 更专业的形式
        for i, (bar, time) in enumerate(zip(bars, total_times)):
            width = bar.get_width()
            text_offset = max(total_times) * 0.05

            # 在柱子内部左侧添加数据
            ax2.text(width - text_offset, bar.get_y() + bar.get_height() / 2,
                     f'{width:.1f}s',
                     va='center', ha='right',
                     color='white', fontsize=30, weight='bold',
                     family='Times New Roman')  # 添加字体设置

            # 在柱子右侧添加数据标签
            ax2.text(width + text_offset / 2, bar.get_y() + bar.get_height() / 2,
                     self.labels[i],
                     va='center', ha='left', fontsize=32,
                     family='Times New Roman')  # 添加字体设置

            # 添加时间差提示
            if i == 1 and len(total_times) == 2:
                ax2.text(width + text_offset / 2, bar.get_y() + bar.get_height() / 2 - 0.15,
                         f'({diff_sign}{diff_percent:.1f}%)',
                         va='center', ha='left', fontsize=28,
                         color='#c44e52' if time_diff > 0 else '#55a868',
                         family='Times New Roman')  # 添加字体设置

        # 美学优化
        ax2.set_facecolor('#f8f8f8')
        ax2.grid(axis='x', linestyle='--', alpha=0.7)
        ax2.set_xlabel("Total Time (s)", fontsize=34, labelpad=15, family='Times New Roman')
        ax2.set_title("Computational Efficiency", pad=20, fontsize=38, family='Times New Roman', fontweight='bold')

        # 隐藏Y轴标签
        ax2.set_yticks([])

        # 设置合适的X轴范围
        max_time = max(total_times)
        ax2.set_xlim(0, max_time * 1.35)

        # 保存为svg格式（矢量图）
        plt.savefig('comparison_professional.svg', dpi=300, bbox_inches='tight')
        plt.show()

    def generate_statistics(self):
        """生成并打印统计信息"""
        stats = {}
        for idx, (result, label) in enumerate(zip(self.results, self.labels)):
            stats[label] = {}
            for algo in result:
                if algo in result:
                    stats[label][algo] = {
                        'final_acc': result[algo]['acc'][-1],
                        'avg_acc': np.mean(result[algo]['acc'][-10:]),  # 最后10轮平均
                        'min_acc': np.min(result[algo]['acc']),
                        'max_acc': np.max(result[algo]['acc']),
                        'avg_time': np.mean(result[algo]['time']),
                        'total_time': np.sum(result[algo]['time']),
                        'avg_comm': np.mean(result[algo]['comm']),
                        'total_comm': np.sum(result[algo]['comm'])
                    }
        return stats

    def print_statistics(self):
        """打印格式化的统计信息"""
        stats = self.generate_statistics()

        print("\n===== Experimental Results Comparison =====")
        for label, algos in stats.items():
            print(f"\n--- {label} ---")
            for algo, values in algos.items():
                print(f"{algo.upper()}:")
                print(f"  Final Accuracy: {values['final_acc']:.4f}")
                print(f"  Average Accuracy (last 10 rounds): {values['avg_acc']:.4f}")
                print(f"  Time: {values['avg_time']:.3f}s/round | Total: {values['total_time']:.1f}s")
                print(f"  Comm: {values['avg_comm']:.2f}MB/round | Total: {values['total_comm']:.2f}MB")

    def visualize_selection_pattern(self, idx=0, num_users=10):
        """
        可视化用户选择模式（仅适用于第一个结果）

        注意：该可视化需要实验结果中有记录用户选择次数
        """
        if 'selection_counts' not in self.results[idx]:
            print("Warning: No selection counts recorded in results.")
            return

        counts = self.results[idx]['selection_counts'][:num_users]

        plt.figure(figsize=(12, 6))
        user_ids = np.arange(len(counts))
        plt.bar(user_ids, counts)
        plt.xlabel("User ID")
        plt.ylabel("Selection Frequency")
        plt.title(f"User Selection Pattern ({self.labels[idx]})")
        plt.savefig('selection_pattern.png')
        plt.show()


if __name__ == "__main__":
    # 定义要比较的两个实验结果文件和对应的标签
    result_files = [
        'experiment_results.pt',  # 原始HSFL实验结果
        'secure_hsfl_results.pt'  # 安全增强的HSFL实验结果
    ]

    labels = [
        'Baseline HSFL',
        'Secure HSFL'
    ]

    # 创建分析器
    analyzer = ResultAnalyzer(result_files, labels)

    # 生成并打印统计信息
    analyzer.print_statistics()

    # 生成对比图表（将保存为PDF矢量图）
    analyzer.plot_accuracy_curves()