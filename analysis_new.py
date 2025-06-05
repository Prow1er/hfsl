import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


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

        sns.set_style("whitegrid")
        plt.rcParams['font.size'] = 15
        self.colors = plt.cm.tab10(np.linspace(0, 1, len(self.results)))

    def plot_accuracy_curves(self, algorithms=['hsfl']):
        """绘制准确率曲线，支持多条曲线在同一图中"""
        plt.figure(figsize=(32, 18))
        plt.rcParams['font.size'] = 20
        plt.rcParams['axes.titlesize'] = 22
        plt.rcParams['axes.labelsize'] = 20

        plt.subplots_adjust(left=0.07, right=0.93, bottom=0.1, top=0.95, hspace=0.35, wspace=0.15)

        ax1 = plt.subplot(2, 1, 1)
        for idx, (result, label) in enumerate(zip(self.results, self.labels)):
            for algo in algorithms:
                if algo in result:
                    acc = np.array(result[algo]['acc'])
                    x = np.arange(len(acc))
                    plt.plot(x, acc,
                             label=f'{label} ({algo.upper()})',
                             color=self.colors[idx],
                             linewidth=2)

        ax1.set_xlabel("Communication Rounds")
        ax1.set_ylabel("Test Accuracy")
        ax1.set_title("(a) Learning Performance Comparison", y=-0.125)
        ax1.legend()

        ax2 = plt.subplot(2, 1, 2)
        for idx, (result, label) in enumerate(zip(self.results, self.labels)):
            for algo in algorithms:
                if algo in result:
                    time = np.array(result[algo]['time'])
                    x = np.arange(len(time))
                    # 使用滑动窗口平均平滑曲线
                    window_size = 10
                    if len(time) > window_size:
                        time_smooth = np.convolve(time, np.ones(window_size) / window_size, mode='valid')
                        plt.plot(x[window_size - 1:], time_smooth,
                                 label=f'{label} ({algo.upper()})',
                                 color=self.colors[idx],
                                 linewidth=2)
                    else:
                        plt.plot(x, time,
                                 label=f'{label} ({algo.upper()})',
                                 color=self.colors[idx],
                                 linewidth=2)

        ax2.set_xlabel("Communication Rounds")
        ax2.set_ylabel("Time per Round (s)")
        ax2.set_title("(b) Computational Efficiency Comparison", y=-0.125)
        ax2.legend()
        plt.savefig('comparison.png')
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
        'experiment_results1.pt',  # 原始HSFL实验结果
        'secure_hsfl_results1.pt'  # 安全增强的HSFL实验结果
    ]

    labels = [
        'Baseline HSFL',
        'Secure HSFL'
    ]

    # 创建分析器
    analyzer = ResultAnalyzer(result_files, labels)

    # 生成并打印统计信息
    analyzer.print_statistics()

    # 生成对比图表
    analyzer.plot_accuracy_curves()
    # analyzer.plot_communication_cost()
    analyzer.plot_time_comparison()
