import torch
from torch.utils.data import Dataset
import numpy as np
import itertools
from tqdm import tqdm

# 使用动态规划算法(Held-Karp算法)求解TSP问题的最优路径
def tsp_opt(points):
    """
    动态规划求解TSP问题 - 时间复杂度O(2^n*n^2)
    来源: https://gist.github.com/mlalevic/6222750

    参数:
    points: 点列表，每个点是(x, y)坐标

    返回:
    最优路径，表示为点索引的数组
    """

    def length(x_coord, y_coord):
        """
        计算两点之间的欧氏距离
        
        参数:
        x_coord: 第一个点的坐标
        y_coord: 第二个点的坐标
        
        返回:
        两点间距离
        """
        return np.linalg.norm(np.asarray(x_coord) - np.asarray(y_coord))

    # Calculate all lengths
    all_distances = [[length(x, y) for y in points] for x in points]
    # Initial value - just distance from 0 to every other point + keep the track of edges
    A = {(frozenset([0, idx+1]), idx+1): (dist, [0, idx+1]) for idx, dist in enumerate(all_distances[0][1:])}
    cnt = len(points)
    for m in range(2, cnt):
        B = {}
        for S in [frozenset(C) | {0} for C in itertools.combinations(range(1, cnt), m)]:
            for j in S - {0}:
                # This will use 0th index of tuple for ordering, the same as if key=itemgetter(0) used
                B[(S, j)] = min([(A[(S-{j}, k)][0] + all_distances[k][j], A[(S-{j}, k)][1] + [j])
                                 for k in S if k != 0 and k != j])
        A = B
    res = min([(A[d][0] + all_distances[0][d[1]], A[d][1]) for d in iter(A)])
    return np.asarray(res[1])

# 随机生成指定数量的TSP实例，每个实例包含一组随机2D坐标点
class TSPDataset(Dataset):
    """
    随机生成TSP问题的数据集
    继承自PyTorch的Dataset类，用于数据加载器
    """

    def __init__(self, data_size, seq_len, solver=tsp_opt, solve=True):
        """
        初始化TSP数据集
        
        参数:
        data_size: 数据集大小（样本数量）
        seq_len: 序列长度（每个TSP问题中的城市数量）
        solver: 解决TSP问题的函数，默认使用tsp_opt动态规划
        solve: 是否在创建数据集时就计算最优解，默认为True
        """
        self.data_size = data_size
        self.seq_len = seq_len
        self.solve = solve
        self.solver = solver
        # 生成数据
        self.data = self._generate_data()

    def __len__(self):
        """
        返回数据集大小
        用于len(dataset)操作
        """
        return self.data_size

    def __getitem__(self, idx):
        """
        获取数据集中的一个样本
        
        参数:
        idx: 样本索引
        
        返回:
        包含点坐标和解决方案的字典
        """
        tensor = torch.from_numpy(self.data['Points_List'][idx]).float()
        solution = torch.from_numpy(self.data['Solutions'][idx]).long() if self.solve else None

        sample = {'Points':tensor, 'Solution':solution}

        return sample

    def _generate_data(self):
        """
        :return: Set of points_list ans their One-Hot vector solutions
        """
        points_list = []
        solutions = []
        data_iter = tqdm(range(self.data_size), unit='data')
        for i, _ in enumerate(data_iter):
            data_iter.set_description('Data points %i/%i' % (i+1, self.data_size))
            points_list.append(np.random.random((self.seq_len, 2)))
        solutions_iter = tqdm(points_list, unit='solve')
        if self.solve:
            for i, points in enumerate(solutions_iter):
                solutions_iter.set_description('Solved %i/%i' % (i+1, len(points_list)))
                solutions.append(self.solver(points))
        else:
            solutions = None

        return {'Points_List':points_list, 'Solutions':solutions}

    def _to1hotvec(self, points):
        """
        :param points: List of integers representing the points indexes
        :return: Matrix of One-Hot vectors
        """
        vec = np.zeros((len(points), self.seq_len))
        for i, v in enumerate(vec):
            v[points[i]] = 1

        return vec
