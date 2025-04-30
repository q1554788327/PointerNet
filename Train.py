"""
Pytorch implementation of Pointer Network.
http://arxiv.org/pdf/1506.03134v1.pdf.
"""

# 导入必要的库
import torch               # PyTorch深度学习框架
import torch.optim as optim  # 优化器
import torch.backends.cudnn as cudnn  # CUDA加速
from torch.autograd import Variable  # 自动求导
from torch.utils.data import DataLoader  # 数据加载器

import numpy as np         # 数值计算库
import argparse            # 命令行参数解析
from tqdm import tqdm      # 进度条显示

# 导入自定义模块
from PointerNet import PointerNet  # Pointer Network模型
from Data_Generator import TSPDataset  # TSP数据集生成器
# 导入模型保存所需模块
import os

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description="Pytorch implementation of Pointer-Net")

# 数据相关参数
parser.add_argument('--train_size', default=1000, type=int, help='训练数据大小') # 10万
parser.add_argument('--val_size', default=100, type=int, help='验证数据大小')
parser.add_argument('--test_size', default=1000, type=int, help='测试数据大小')
parser.add_argument('--batch_size', default=256, type=int, help='批次大小')

# 训练相关参数
parser.add_argument('--nof_epoch', default=50, type=int, help='训练轮数')
parser.add_argument('--lr', type=float, default=0.0001, help='学习率')

# GPU相关参数
parser.add_argument('--gpu', default=True, action='store_true', help='是否启用GPU')

# TSP问题参数
parser.add_argument('--nof_points', type=int, default=5, help='TSP问题中的点数量')

# 网络结构参数
parser.add_argument('--embedding_size', type=int, default=128, help='嵌入维度')
parser.add_argument('--hiddens', type=int, default=512, help='隐藏单元数量')
parser.add_argument('--nof_lstms', type=int, default=2, help='LSTM层数')
parser.add_argument('--dropout', type=float, default=0., help='Dropout比例')
parser.add_argument('--bidir', default=True, action='store_true', help='是否使用双向LSTM')

# 解析命令行参数
params = parser.parse_args()

# 检查GPU可用性
if params.gpu and torch.cuda.is_available():
    USE_CUDA = True
    print('Using GPU, %i devices.' % torch.cuda.device_count())
else:
    USE_CUDA = False

# 初始化Pointer Network模型
model = PointerNet(
    params.embedding_size,  # 嵌入维度
    params.hiddens,         # 隐藏单元数量
    params.nof_lstms,       # LSTM层数
    params.dropout,         # Dropout比例
    params.bidir            # 是否使用双向LSTM
)

# 创建TSP数据集
dataset = TSPDataset(
    params.train_size,      # 训练数据大小
    params.nof_points       # TSP问题中的点数量
)

# 创建数据加载器，用于批量加载数据
dataloader = DataLoader(
    dataset,                # 数据集
    batch_size=params.batch_size,  # 批次大小
    shuffle=True,           # 随机打乱数据
    num_workers=4           # 并行加载进程数
)

# 如果启用GPU，将模型转移到GPU
if USE_CUDA:
    model.cuda()
    # 使用DataParallel在多个GPU上并行运算
    net = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    # 优化CUDA性能
    cudnn.benchmark = True

# 定义损失函数 - 交叉熵损失
CCE = torch.nn.CrossEntropyLoss()

# 定义优化器 - Adam优化器
# 仅更新需要梯度的参数
model_optim = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=params.lr            # 学习率
)

# 记录训练过程中的损失
losses = []

# 开始训练循环
for epoch in range(params.nof_epoch):
    batch_loss = []  # 每个epoch的批次损失
    iterator = tqdm(dataloader, unit='Batch')  # 带进度条的数据迭代器

    # 遍历数据批次
    for i_batch, sample_batched in enumerate(iterator):
        # 更新进度条描述
        iterator.set_description('Batch %i/%i' % (epoch+1, params.nof_epoch))

        # 获取输入和目标
        train_batch = Variable(sample_batched['Points'])  # 城市坐标
        target_batch = Variable(sample_batched['Solution'])  # 最优路径

        # 如果使用GPU，将数据转移到GPU
        if USE_CUDA:
            train_batch = train_batch.cuda()
            target_batch = target_batch.cuda()

        # 前向传播
        o, p = model(train_batch)  # o: 输出概率, p: 预测的指针
        
        # 调整输出形状以匹配损失函数要求
        o = o.contiguous().view(-1, o.size()[-1])
        # 调整目标形状
        target_batch = target_batch.view(-1)

        # 计算损失
        loss = CCE(o, target_batch)

        # 记录损失值
        losses.append(loss.item())  # 注：新版PyTorch中应改为loss.item()
        batch_loss.append(loss.item())

        # 反向传播和优化
        model_optim.zero_grad()  # 清空梯度
        loss.backward()          # 反向传播
        model_optim.step()       # 更新参数

        # 更新进度条显示的损失值
        iterator.set_postfix(loss='{}'.format(loss.item()))

    # 显示本轮的平均损失
    iterator.set_postfix(loss=np.average(batch_loss))
    
    # 显示本轮的平均损失
    epoch_loss = np.average(batch_loss)
    iterator.set_postfix(loss=epoch_loss)
    
    # 保存模型参数
    if (epoch + 1) % 10 == 0:  # 每10个epoch保存一次
        save_dir = 'checkpoints'
        os.makedirs(save_dir, exist_ok=True)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': model_optim.state_dict(),
            'loss': epoch_loss,
            'params': params,
        }, os.path.join(save_dir, f'tsp{params.nof_points}_epoch{epoch+1}.pt'))
        print(f'模型已保存至 {save_dir}/tsp{params.nof_points}_epoch{epoch+1}.pt')

# 保存最终模型
torch.save({
    'epoch': params.nof_epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': model_optim.state_dict(),
    'loss': np.mean(losses[-len(dataloader):]),  # 最后一个epoch的平均损失
    'params': params,
}, os.path.join('checkpoints', f'tsp{params.nof_points}_final.pt'))
print(f'最终模型已保存至 checkpoints/tsp{params.nof_points}_final.pt')