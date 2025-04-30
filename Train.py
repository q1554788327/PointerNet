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
parser.add_argument('--ensemble_size', type=int, default=5, help='集成的模型数量')

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

# 创建训练数据集
train_dataset = TSPDataset(
    params.train_size,      # 训练数据大小
    params.nof_points       # TSP问题中的点数量
)

# 创建验证数据集
val_dataset = TSPDataset(
    params.val_size,        # 验证数据大小
    params.nof_points       # TSP问题中的点数量
)

# 创建训练数据加载器
train_dataloader = DataLoader(
    train_dataset,          # 数据集
    batch_size=params.batch_size,  # 批次大小
    shuffle=True,           # 随机打乱数据
    num_workers=4           # 并行加载进程数
)

# 创建验证数据加载器
val_dataloader = DataLoader(
    val_dataset,            # 数据集
    batch_size=params.batch_size,  # 批次大小
    shuffle=False,          # 不需要打乱验证数据
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
train_losses = []
val_losses = []

# Checkpoint Ensemble相关变量
save_dir = 'checkpoints'
os.makedirs(save_dir, exist_ok=True)

# 计算保存检查点的间隔
checkpoint_interval = params.nof_epoch // params.ensemble_size
if checkpoint_interval == 0:
    checkpoint_interval = 1
    print(f"警告: ensemble_size ({params.ensemble_size}) 大于训练轮数 ({params.nof_epoch}).")
    print(f"将保存每一轮的模型，共 {params.nof_epoch} 个检查点.")

# 定义验证函数
def validate():
    model.eval()  # 设置为评估模式
    val_loss = []
    with torch.no_grad():  # 不计算梯度
        for sample_batched in tqdm(val_dataloader, desc="Validating", unit="Batch"):
            # 获取输入和目标
            val_batch = Variable(sample_batched['Points'])
            val_target = Variable(sample_batched['Solution'])
            
            if USE_CUDA:
                val_batch = val_batch.cuda()
                val_target = val_target.cuda()
            
            # 前向传播
            o, p = model(val_batch)
            o = o.contiguous().view(-1, o.size()[-1])
            val_target = val_target.view(-1)
            
            # 计算损失
            loss = CCE(o, val_target)
            val_loss.append(loss.item())
    
    model.train()  # 恢复为训练模式
    return np.mean(val_loss)

# 开始训练循环
print("Starting training...")
for epoch in range(params.nof_epoch):
    model.train()  # 确保模型处于训练模式
    batch_loss = []  # 每个epoch的批次损失
    iterator = tqdm(train_dataloader, unit='Batch', desc=f'Epoch {epoch+1}/{params.nof_epoch}')

    # 遍历数据批次
    for i_batch, sample_batched in enumerate(iterator):
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
        train_losses.append(loss.item())
        batch_loss.append(loss.item())

        # 反向传播和优化
        model_optim.zero_grad()  # 清空梯度
        loss.backward()          # 反向传播
        model_optim.step()       # 更新参数

        # 更新进度条显示的损失值
        iterator.set_postfix(train_loss=loss.item())

    # 计算本轮的平均损失
    epoch_train_loss = np.average(batch_loss)
    print(f"Epoch {epoch+1}/{params.nof_epoch}, Train Loss: {epoch_train_loss:.6f}")
    
    # 在验证集上评估
    epoch_val_loss = validate()
    val_losses.append(epoch_val_loss)
    print(f"Epoch {epoch+1}/{params.nof_epoch}, Validation Loss: {epoch_val_loss:.6f}")
    
    # 根据checkpoint_interval保存模型检查点
    if (epoch + 1) % checkpoint_interval == 0 or epoch == params.nof_epoch - 1:
        # 保存检查点
        model_path = os.path.join(save_dir, f'tsp{params.nof_points}_checkpoint_{epoch+1}.pt')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': model_optim.state_dict(),
            'train_loss': epoch_train_loss,
            'val_loss': epoch_val_loss,
            'params': params,
        }, model_path)
        print(f'保存检查点: {model_path}')

# 训练结束提示
print(f'训练完成! 已保存 {params.ensemble_size} 个检查点，用于后续模型集成。')