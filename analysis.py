import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class ResultAnalyzer:
    def __init__(self, result_path):
        self.results = torch.load(result_path)
        sns.set_style("whitegrid")
        plt.rcParams['font.size'] = 15

    def plot_accuracy_curves(self, algorithms=['hsfl']):
        plt.figure(figsize=(10, 6))
        for algo in algorithms:
            acc = np.array(self.results[algo]['acc'])
            x = np.arange(len(acc))
            plt.plot(x, acc, label=algo.upper())

        plt.xlabel("Communication Rounds")
        plt.ylabel("Test Accuracy")
        plt.title("Learning Performance Comparison")
        plt.legend()
        plt.tight_layout()
        plt.savefig('accuracy_comparison.png')
        plt.show()

    def plot_communication_cost(self):
        plt.figure(figsize=(10, 6))
        x = np.arange(len(self.results['hsfl']['comm']))

        plt.plot(x, np.cumsum(self.results['hsfl']['comm']),
                 label='HSFL')

        plt.xlabel("Communication Rounds")
        plt.ylabel("Cumulative Communication Cost (MB)")
        plt.title("Communication Efficiency Comparison")
        plt.legend()
        plt.tight_layout()
        plt.savefig('communication_cost.png')
        plt.show()

    def generate_statistics(self):
        stats = {}
        for algo in ['hsfl']:
            stats[algo] = {
                'final_acc': self.results[algo]['acc'][-1],
                'avg_time': np.mean(self.results[algo]['time']),
                'total_comm': np.sum(self.results[algo]['comm'])
            }
        return stats

    def visualize_selection_pattern(self, num_users=10):
        # 示例：可视化前10个用户的选择频率
        plt.figure(figsize=(12, 6))
        user_ids = np.arange(num_users)
        counts = np.random.rand(num_users)  # 需要实际记录选择次数
        plt.bar(user_ids, counts)
        plt.xlabel("User ID")
        plt.ylabel("Selection Frequency")
        plt.title("User Selection Pattern")
        plt.savefig('selection_pattern.png')
        plt.show()


if __name__ == "__main__":
    analyzer = ResultAnalyzer('experiment_results.pt')  # experiment_results / secure_hsfl_results

    # 生成分析图表
    analyzer.plot_accuracy_curves()
    analyzer.plot_communication_cost()

    # 打印统计信息
    stats = analyzer.generate_statistics()
    print("===== Experimental Results =====")
    for algo in stats:
        print(f"{algo.upper()}:")
        print(f"  Final Accuracy: {stats[algo]['final_acc']:.4f}")
        print(f"  Average Time per Round: {stats[algo]['avg_time']:.2f}s")
        print(f"  Total Communication: {stats[algo]['total_comm']:.2f}MB\n")

    # 可视化用户选择模式（需要记录实际数据）
    analyzer.visualize_selection_pattern()