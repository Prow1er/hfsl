#data_utils.py
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset

# 定义数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST标准化参数
])


class PartitionedDataset(Dataset):
    """封装划分后的数据集"""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

    def __len__(self):
        return len(self.indices)


def load_mnist(data_dir="./data"):
    """加载MNIST数据集"""
    train_set = datasets.MNIST(
        data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(
        data_dir, train=False, download=True, transform=transform)
    return train_set, test_set


def iid_partition(dataset, num_users):
    """IID数据划分"""
    num_items = len(dataset) // num_users
    indices = np.random.permutation(len(dataset))
    return [indices[i * num_items:(i + 1) * num_items] for i in range(num_users)]


def noniid_partition(dataset, num_users, shards_per_user=2, num_classes=10):
    """非IID划分（基于标签分片）"""
    # 按标签排序索引
    labels = np.array(dataset.targets)
    idxs = np.arange(len(labels))
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # 划分分片
    num_shards = num_users * shards_per_user
    shard_size = len(idxs) // num_shards
    shards = [idxs[i * shard_size:(i + 1) * shard_size] for i in range(num_shards)]

    # 分配分片给用户
    np.random.shuffle(shards)
    user_shards = np.array_split(shards, num_users)
    user_indices = [np.concatenate(shards).astype(int) for shards in user_shards]
    return user_indices


def dirichlet_partition(dataset, num_users, alpha=0.1, num_classes=10):
    """Dirichlet分布划分"""
    labels = np.array(dataset.targets)
    idxs = np.arange(len(labels))

    # 按类别组织索引
    class_idxs = [idxs[labels == k] for k in range(num_classes)]

    # 生成Dirichlet分布矩阵
    proportions = np.random.dirichlet(np.repeat(alpha, num_users), num_classes)

    # 分配样本
    user_indices = [[] for _ in range(num_users)]
    for c in range(num_classes):
        np.random.shuffle(class_idxs[c])
        cum_prop = np.cumsum(proportions[c])
        splits = (cum_prop * len(class_idxs[c])).astype(int)[:-1]
        user_samples = np.split(class_idxs[c], splits)

        for u in range(num_users):
            user_indices[u].extend(user_samples[u].tolist())

    # 打乱每个用户的样本顺序
    for u in range(num_users):
        np.random.shuffle(user_indices[u])

    return user_indices


def save_partition(partition, save_path):
    """保存划分结果"""
    torch.save(partition, save_path)


def load_partition(load_path):
    """加载划分结果"""
    return torch.load(load_path)


if __name__ == "__main__":
    # 示例用法
    train_set, test_set = load_mnist()

    # 生成不同划分
    iid_indices = iid_partition(train_set, num_users=100)
    noniid_indices = noniid_partition(train_set, num_users=100)
    dirichlet_indices = dirichlet_partition(train_set, num_users=100, alpha=0.1)

    # 保存划分结果
    save_partition(iid_indices, "iid_partition.pt")
    save_partition(noniid_indices, "noniid_partition.pt")
    save_partition(dirichlet_indices, "dirichlet_partition.pt")
