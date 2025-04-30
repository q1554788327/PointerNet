import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

'''
Pointer Network 是一种序列到序列的神经网络结构，专门设计用来解决输出是输入序列元素排列的问题（如排序、TSP等）。
该模型的主要特点是解码器不生成新元素，而是通过注意力机制"指向"输入序列中的元素。模型结构如下：

编码器 (Encoder)

基于LSTM结构，将输入序列编码为上下文表示
支持单向或双向LSTM配置
处理形状为[batch_size, seq_len, embedding_dim]的输入
注意力机制 (Attention)

计算解码器隐藏状态与编码器输出之间的相关性
生成注意力权重，决定指向输入序列中哪个位置
使用掩码机制避免重复选择相同位置
解码器 (Decoder)

自定义实现的LSTM，集成了注意力机制
在每一步生成指向输入序列元素的概率分布
使用掩码跟踪已选择的位置，防止重复选择
完整网络 (PointerNet)

集成编码器-解码器架构
嵌入层将原始输入转换为高维表示
处理单/双向LSTM的转换逻辑
'''


class Encoder(nn.Module):
    """
    Encoder类 - Pointer Network的编码器部分
    作用：将输入序列编码成隐藏表示，使用LSTM网络
    """

    def __init__(self, embedding_dim,
                 hidden_dim,
                 n_layers,
                 dropout,
                 bidir):
        """
        初始化编码器

        参数:
        embedding_dim: 嵌入维度，输入特征的维度
        hidden_dim: LSTM隐藏状态的维度
        n_layers: LSTM的层数
        dropout: dropout比例（0-1之间）
        bidir: 是否使用双向LSTM
        """

        super(Encoder, self).__init__()
        # 如果是双向LSTM，隐藏维度减半（因为前向和后向的结果会拼接）
        self.hidden_dim = hidden_dim//2 if bidir else hidden_dim
        # 如果是双向LSTM，层数翻倍（因为每层包含前向和后向）
        self.n_layers = n_layers*2 if bidir else n_layers
        self.bidir = bidir
        # 创建LSTM层
        self.lstm = nn.LSTM(embedding_dim,
                            self.hidden_dim,
                            n_layers,
                            dropout=dropout,
                            bidirectional=bidir)

        # 创建零张量参数，用于初始化隐藏状态并支持GPU运算
        self.h0 = Parameter(torch.zeros(1), requires_grad=False)
        self.c0 = Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, embedded_inputs,
                hidden):
        """
        编码器前向传播

        参数:
        embedded_inputs: Pointer-Net的嵌入输入，形状为[batch_size, seq_len, embedding_dim]
        hidden: LSTM的初始隐藏状态 (h, c)

        返回:
        outputs: LSTM的所有时间步输出
        hidden: 最终的隐藏状态 (h, c)
        """

        # 转换维度顺序为[seq_len, batch_size, embedding_dim]，因为PyTorch LSTM要求这种格式
        embedded_inputs = embedded_inputs.permute(1, 0, 2)

        # 通过LSTM处理序列
        outputs, hidden = self.lstm(embedded_inputs, hidden)

        # 将输出转换回[batch_size, seq_len, hidden_dim]格式
        return outputs.permute(1, 0, 2), hidden

    def init_hidden(self, embedded_inputs):
        """
        初始化隐藏状态

        参数:
        embedded_inputs: 嵌入后的输入，用于获取batch_size

        返回:
        (h0, c0): LSTM的初始隐藏状态和细胞状态
        """

        batch_size = embedded_inputs.size(0)

        # 调整张量形状并复制到所需维度
        # 形状: [n_layers, batch_size, hidden_dim]
        h0 = self.h0.unsqueeze(0).unsqueeze(0).repeat(self.n_layers,
                                                      batch_size,
                                                      self.hidden_dim)
        c0 = self.h0.unsqueeze(0).unsqueeze(0).repeat(self.n_layers,
                                                      batch_size,
                                                      self.hidden_dim)

        return h0, c0


class Attention(nn.Module):
    """
    注意力机制模块 - Pointer Network使用的注意力机制
    作用：计算注意力权重，使解码器能够关注输入序列中的特定位置
    """
    
    def __init__(self, input_dim,
                 hidden_dim):
        """
        初始化注意力模块

        参数:
        input_dim: 输入特征的维度
        hidden_dim: 注意力机制中隐藏单元的数量
        """

        super(Attention, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 输入线性变换层
        self.input_linear = nn.Linear(input_dim, hidden_dim)
        # 上下文线性变换层（使用1x1卷积）
        self.context_linear = nn.Conv1d(input_dim, hidden_dim, 1, 1)
        # 注意力向量V，可学习参数
        self.V = Parameter(torch.FloatTensor(hidden_dim), requires_grad=True)
        # 负无穷常量，用于掩码操作
        self._inf = Parameter(torch.FloatTensor([float('-inf')]), requires_grad=False)
        # 激活函数
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

        # 初始化向量V为均匀分布
        nn.init.uniform(self.V, -1, 1)

    def forward(self, input,
                context,
                mask):
        """
        注意力机制前向传播

        参数:
        input: 隐藏状态h，形状为[batch_size, hidden_dim]
        context: 注意力上下文，即编码器输出，形状为[batch_size, seq_len, hidden_dim]
        mask: 选择掩码，用于屏蔽已选择的位置，形状为[batch_size, seq_len]

        返回:
        hidden_state: 加权上下文向量，形状为[batch_size, hidden_dim]
        alpha: 注意力权重，形状为[batch_size, seq_len]
        """

        # 变换输入并扩展维度：[batch_size, hidden_dim] -> [batch_size, hidden_dim, seq_len]
        inp = self.input_linear(input).unsqueeze(2).expand(-1, -1, context.size(1))

        # 转置上下文并应用变换：[batch_size, seq_len, hidden_dim] -> [batch_size, hidden_dim, seq_len]
        context = context.permute(0, 2, 1)
        ctx = self.context_linear(context)

        # 扩展V向量维度以进行批处理计算：[hidden_dim] -> [batch_size, 1, hidden_dim]
        V = self.V.unsqueeze(0).expand(context.size(0), -1).unsqueeze(1)

        # 计算注意力分数：[batch_size, seq_len]
        att = torch.bmm(V, self.tanh(inp + ctx)).squeeze(1)
        # 对已选择的位置应用掩码（设为负无穷）
        if len(att[mask]) > 0:
            att[mask] = self.inf[mask]
        # 应用softmax得到注意力权重
        alpha = self.softmax(att)

        # 使用注意力权重计算加权上下文向量
        hidden_state = torch.bmm(ctx, alpha.unsqueeze(2)).squeeze(2)

        return hidden_state, alpha

    def init_inf(self, mask_size):
        """
        初始化负无穷张量用于掩码操作
        
        参数:
        mask_size: 掩码的大小
        """
        self.inf = self._inf.unsqueeze(1).expand(*mask_size)


class Decoder(nn.Module):
    """
    解码器模型 - Pointer Network的解码部分
    作用：基于编码器的输出和注意力机制生成指针序列
    """

    def __init__(self, embedding_dim,
                 hidden_dim):
        """
        初始化解码器

        参数:
        embedding_dim: Pointer-Net中嵌入的维度
        hidden_dim: 解码器RNN的隐藏单元数量
        """

        super(Decoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # 定义LSTM门控所需的线性变换
        self.input_to_hidden = nn.Linear(embedding_dim, 4 * hidden_dim)
        self.hidden_to_hidden = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.hidden_out = nn.Linear(hidden_dim * 2, hidden_dim)
        self.att = Attention(hidden_dim, hidden_dim)

        # 用于掩码操作的参数（同时支持GPU运算）
        self.mask = Parameter(torch.ones(1), requires_grad=False)
        self.runner = Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, embedded_inputs,
                decoder_input,
                hidden,
                context):
        """
        解码器前向传播

        参数:
        embedded_inputs: Pointer-Net的嵌入输入，形状为[batch_size, seq_len, embedding_dim]
        decoder_input: 解码器的初始输入，形状为[batch_size, embedding_dim]
        hidden: 解码器的初始隐藏状态，来自编码器的最终状态
        context: 编码器的输出，形状为[batch_size, seq_len, hidden_dim]

        返回:
        (outputs, pointers): 输出概率和指针索引
        hidden: 最终隐藏状态
        """

        batch_size = embedded_inputs.size(0)
        input_length = embedded_inputs.size(1)

        # (batch, seq_len)
        mask = self.mask.repeat(input_length).unsqueeze(0).repeat(batch_size, 1)
        self.att.init_inf(mask.size())

        # Generating arang(input_length), broadcasted across batch_size
        runner = self.runner.repeat(input_length)
        for i in range(input_length):
            runner.data[i] = i
        runner = runner.unsqueeze(0).expand(batch_size, -1).long()

        outputs = []
        pointers = []

        def step(x, hidden):
            """
            递归步骤函数 - 实现LSTM单元和注意力计算

            参数:
            x: t时刻的输入，形状为[batch_size, embedding_dim]
            hidden: t-1时刻的隐藏状态 (h, c)

            返回:
            hidden_t: 更新后的隐藏状态h
            c_t: 更新后的细胞状态c
            output: 注意力输出（概率分布）
            """

            # Regular LSTM
            h, c = hidden

            gates = self.input_to_hidden(x) + self.hidden_to_hidden(h)
            input, forget, cell, out = gates.chunk(4, 1)

            input = F.sigmoid(input)
            forget = F.sigmoid(forget)
            cell = F.tanh(cell)
            out = F.sigmoid(out)

            c_t = (forget * c) + (input * cell)
            h_t = out * F.tanh(c_t)

            # Attention section
            hidden_t, output = self.att(h_t, context, torch.eq(mask, 0))
            hidden_t = F.tanh(self.hidden_out(torch.cat((hidden_t, h_t), 1)))

            return hidden_t, c_t, output

        # Recurrence loop
        for _ in range(input_length):
            h_t, c_t, outs = step(decoder_input, hidden)
            hidden = (h_t, c_t)

            # Masking selected inputs
            masked_outs = outs * mask

            # Get maximum probabilities and indices
            max_probs, indices = masked_outs.max(1)
            one_hot_pointers = (runner == indices.unsqueeze(1).expand(-1, outs.size()[1])).float()

            # Update mask to ignore seen indices
            mask  = mask * (1 - one_hot_pointers)

            # Get embedded inputs by max indices
            embedding_mask = one_hot_pointers.unsqueeze(2).expand(-1, -1, self.embedding_dim).byte()
            decoder_input = embedded_inputs[embedding_mask.data].view(batch_size, self.embedding_dim)

            outputs.append(outs.unsqueeze(0))
            pointers.append(indices.unsqueeze(1))

        outputs = torch.cat(outputs).permute(1, 0, 2)
        pointers = torch.cat(pointers, 1)

        return (outputs, pointers), hidden


class PointerNet(nn.Module):
    """
    Pointer Network模型
    作用：将输入序列映射为指向序列中元素的指针序列，用于解决排序、TSP等问题
    """

    def __init__(self, embedding_dim,
                 hidden_dim,
                 lstm_layers,
                 dropout,
                 bidir=False):
        """
        初始化Pointer Network

        参数:
        embedding_dim: 嵌入维度
        hidden_dim: 编码器隐藏单元数量
        lstm_layers: LSTM的层数
        dropout: dropout比例（0-1之间）
        bidir: 是否使用双向LSTM
        """

        super(PointerNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.bidir = bidir
        
        # 输入嵌入层，假设输入为2维坐标
        self.embedding = nn.Linear(2, embedding_dim)
        
        # 创建编码器
        self.encoder = Encoder(embedding_dim,
                               hidden_dim,
                               lstm_layers,
                               dropout,
                               bidir)
        
        # 创建解码器
        self.decoder = Decoder(embedding_dim, hidden_dim)
        
        # 解码器的初始输入，可学习参数
        self.decoder_input0 = Parameter(torch.FloatTensor(embedding_dim), requires_grad=False)

        # 初始化解码器初始输入为均匀分布
        nn.init.uniform(self.decoder_input0, -1, 1)

    def forward(self, inputs):
        """
        Pointer Network前向传播

        参数:
        inputs: 输入序列，形状为[batch_size, seq_len, 2]（假设每个元素是2D点）

        返回:
        outputs: 指针概率分布
        pointers: 预测的指针索引
        """

        batch_size = inputs.size(0)
        input_length = inputs.size(1)

        # 扩展解码器的初始输入到批次大小
        decoder_input0 = self.decoder_input0.unsqueeze(0).expand(batch_size, -1)

        # 展平输入并应用嵌入
        inputs = inputs.view(batch_size * input_length, -1)
        embedded_inputs = self.embedding(inputs).view(batch_size, input_length, -1)

        # 初始化编码器隐藏状态并运行编码器
        encoder_hidden0 = self.encoder.init_hidden(embedded_inputs)
        encoder_outputs, encoder_hidden = self.encoder(embedded_inputs, encoder_hidden0)
        
        # 根据是否使用双向LSTM，准备解码器的初始隐藏状态
        if self.bidir:
            # 修复：处理不同PyTorch版本的隐藏状态格式差异
            if isinstance(encoder_hidden[0], tuple):
                # 旧版本格式：元组
                decoder_hidden0 = (torch.cat(encoder_hidden[0][-2:], dim=-1),
                                torch.cat(encoder_hidden[1][-2:], dim=-1))
            else:
                # 新版本格式：张量
                # 获取最后一层双向LSTM的前向和后向隐藏状态
                h_n, c_n = encoder_hidden
                # h_n形状: [num_layers*num_directions, batch_size, hidden_dim]
                # 我们需要获取最后两个状态（前向和后向的最后层）并拼接
                last_fw = h_n[-2]  # 前向最后层
                last_bw = h_n[-1]  # 后向最后层
                last_c_fw = c_n[-2]
                last_c_bw = c_n[-1]
                
                decoder_hidden0 = (torch.cat([last_fw, last_bw], dim=-1),
                                torch.cat([last_c_fw, last_c_bw], dim=-1))
        else:
            # 如果是单向LSTM，直接使用最后的隐藏状态
            decoder_hidden0 = (encoder_hidden[0][-1], encoder_hidden[1][-1])
        
        # 运行解码器
        (outputs, pointers), decoder_hidden = self.decoder(embedded_inputs,
                                                        decoder_input0,
                                                        decoder_hidden0,
                                                        encoder_outputs)

        return outputs, pointers