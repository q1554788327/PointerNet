import torch
import numpy as np
import matplotlib.pyplot as plt
from Data_Generator import TSPDataset, tsp_opt

def test_tsp_solver():
    """测试TSP求解器是否能正确计算简单案例的最优路径"""
    # 创建一个简单的TSP问题（4个点的正方形）
    points = np.array([
        [0, 0],
        [0, 1],
        [1, 1],
        [1, 0]
    ])
    
    # 最优路径应该是围绕正方形的闭环
    solution = tsp_opt(points)
    print("Simple square TSP solution:", solution)
    
    # 计算路径长度
    total_length = 0
    for i in range(len(solution)-1):
        total_length += np.linalg.norm(points[solution[i]] - points[solution[i+1]])
    # 闭环回到起点
    total_length += np.linalg.norm(points[solution[-1]] - points[solution[0]])
    print(f"Path length: {total_length:.4f} (should be close to 4.0 for a unit square)")

def test_dataset_generation():
    """测试数据集生成功能"""
    # 生成一个小型数据集
    print("Generating small dataset...")
    small_dataset = TSPDataset(data_size=10, seq_len=10, solve=True)
    
    # 检查数据集大小
    assert len(small_dataset) == 10, "Dataset size doesn't match expected value"
    
    # 获取第一个样本
    sample = small_dataset[0]
    print("Sample points shape:", sample['Points'].shape)
    print("Sample solution shape:", sample['Solution'].shape)
    
    # 验证解决方案有效性
    points = sample['Points'].numpy()
    solution = sample['Solution'].numpy()
    print("First sample solution:", solution)
    
    return small_dataset, sample

def visualize_tsp_solution(points, solution, save_path='tsp_solution.png'):
    """
    可视化TSP问题及其解决方案并保存为图片
    
    参数:
    points: 点坐标数组
    solution: 解决方案路径索引
    save_path: 图片保存路径
    """
    plt.figure(figsize=(8, 8))
    
    # 绘制点
    plt.scatter(points[:, 0], points[:, 1], c='blue', s=50)
    
    # 为每个点添加索引标签
    for i, (x, y) in enumerate(points):
        plt.annotate(str(i), (x, y), textcoords="offset points", 
                    xytext=(0, 5), ha='center')
    
    # 绘制路径
    for i in range(len(solution)-1):
        plt.plot([points[solution[i], 0], points[solution[i+1], 0]],
                 [points[solution[i], 1], points[solution[i+1], 1]], 'r-')
    
    # 连接最后一个点和第一个点，形成闭环
    plt.plot([points[solution[-1], 0], points[solution[0], 0]],
             [points[solution[-1], 1], points[solution[0], 1]], 'r-')
    
    plt.title("TSP Solution Visualization")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.grid(True)
    
    # 保存图片而不是显示
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"TSP solution visualization saved to {save_path}")

if __name__ == "__main__":
    # 测试TSP求解器
    print("===== Testing TSP Solver =====")
    test_tsp_solver()
    
    # 测试数据集生成
    print("\n===== Testing Dataset Generation =====")
    dataset, sample = test_dataset_generation()
    
    # 可视化一个TSP解决方案
    print("\n===== Visualizing TSP Solution =====")
    visualize_tsp_solution(sample['Points'].numpy(), sample['Solution'].numpy())
    
    print("\nAll tests completed!")