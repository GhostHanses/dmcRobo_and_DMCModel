from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math


class ContentEncoder(nn.Module):
    """专门用于编码消息内容的模块"""

    def __init__(self, d_model=768, nhead=8, num_layers=2, max_seq_len=512):
        super().__init__()
        from transformers import AutoTokenizer, AutoModel

        # 加载BERT模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        self.bert = AutoModel.from_pretrained("bert-base-chinese")

        # 冻结BERT参数（可选）
        for param in self.bert.parameters():
            param.requires_grad = False

        # BERT的输出维度
        bert_hidden_size = self.bert.config.hidden_size

        # 如果BERT的隐藏层大小与d_model不同，需要投影层
        if bert_hidden_size != d_model:
            self.projection = nn.Linear(bert_hidden_size, d_model)
        else:
            self.projection = nn.Identity()

        # 用于进一步聚合文本序列的Transformer编码器
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=1024,
            batch_first=True,
            dropout=0.1
        )
        self.sequence_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 注意力池化层，将序列聚合为单个向量
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )

        # 最终融合层
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # 注意力池化 + 平均池化
            nn.ReLU(),
            nn.LayerNorm(d_model)
        )

    def forward(self, text_content):
        """
        编码单个消息的文本内容
        text_content: 字符串或字符串列表
        返回: [batch_size, d_model]
        """
        if isinstance(text_content, str):
            text_content = [text_content]

        # 使用BERT获取token级别的表示
        inputs = self.tokenizer(
            text_content,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        # 将输入移动到正确的设备
        device = next(self.bert.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():  # BERT被冻结
            bert_outputs = self.bert(**inputs)

        # 获取BERT的最后一层隐藏状态 [batch_size, seq_len, bert_hidden_size]
        token_embeddings = bert_outputs.last_hidden_state

        # 投影到目标维度
        token_embeddings = self.projection(token_embeddings)

        # 通过序列编码器进一步处理
        # 创建注意力mask（忽略padding tokens）
        attention_mask = inputs['attention_mask'].bool()
        encoded_sequence = self.sequence_encoder(
            token_embeddings,
            src_key_padding_mask=~attention_mask
        )

        # 应用注意力mask
        encoded_sequence = encoded_sequence * attention_mask.unsqueeze(-1)

        # 方法1: 注意力池化
        attention_weights = self.attention_pool(encoded_sequence)  # [batch_size, seq_len, 1]
        attention_weights = attention_weights.masked_fill(
            ~attention_mask.unsqueeze(-1), float('-inf')
        )
        attention_weights = F.softmax(attention_weights, dim=1)

        # 加权平均
        attended_vector = torch.sum(attention_weights * encoded_sequence, dim=1)  # [batch_size, d_model]

        # 方法2: 平均池化（作为补充）
        sum_embeddings = torch.sum(encoded_sequence * attention_mask.unsqueeze(-1), dim=1)
        seq_lengths = attention_mask.sum(dim=1, keepdim=True).float()
        mean_vector = sum_embeddings / seq_lengths.clamp(min=1.0)

        # 融合两种池化方式的结果
        combined = torch.cat([attended_vector, mean_vector], dim=-1)
        final_vector = self.fusion(combined)  # [batch_size, d_model]

        return final_vector


class PropertyAggregator(nn.Module):
    """聚合单个消息的多个属性"""

    def __init__(self, num_properties=3, d_model=768, nhead=8, num_layers=1):
        super().__init__()
        self.num_properties = num_properties

        # 每个属性先投影到统一的维度
        self.property_projections = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_properties)
        ])

        # 用于属性间交互的Transformer
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=1024,
            batch_first=True,
            dropout=0.1
        )
        self.property_interaction = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 属性融合层
        self.property_fusion = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(d_model)
        )

        # 属性重要性权重（可学习）
        self.property_weights = nn.Parameter(torch.ones(num_properties))

    def forward(self, properties_embeddings):
        """
        properties_embeddings: [batch_size, num_msgs, num_properties, d_model]
        输出: [batch_size, num_msgs, d_model]
        """
        batch_size, num_msgs, num_props, d_model = properties_embeddings.shape

        # 对每个属性应用投影
        projected_properties = []
        for prop_idx in range(num_props):
            prop_emb = properties_embeddings[:, :, prop_idx, :]  # [batch_size, num_msgs, d_model]
            projected = self.property_projections[prop_idx](prop_emb)
            projected_properties.append(projected)

        # 堆叠属性 [batch_size, num_msgs, num_props, d_model]
        all_properties = torch.stack(projected_properties, dim=2)

        # 对每个消息的属性进行聚合
        aggregated_msgs = []
        for msg_idx in range(num_msgs):
            # 获取当前消息的所有属性 [batch_size, num_props, d_model]
            msg_properties = all_properties[:, msg_idx, :, :]

            # 通过Transformer进行属性间交互
            interacted_properties = self.property_interaction(msg_properties)

            # 计算属性权重（softmax归一化）
            weights = F.softmax(self.property_weights, dim=0)
            weights = weights.view(1, -1, 1)  # [1, num_props, 1]

            # 加权融合
            weighted_sum = torch.sum(interacted_properties * weights, dim=1)  # [batch_size, d_model]

            # 最终融合
            fused = self.property_fusion(weighted_sum)  # [batch_size, d_model]
            aggregated_msgs.append(fused)

        # 堆叠所有消息 [batch_size, num_msgs, d_model]
        return torch.stack(aggregated_msgs, dim=1)


class ExtendedTransformerEncoderLayer(nn.Module):
    """扩展的Transformer编码器层，支持邻接矩阵干预"""

    def __init__(self, d_model=768, nhead=8, dim_feedforward=1024, dropout=0.1):
        super().__init__()

        # 自注意力机制
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # 前馈网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 归一化层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

        # 可学习的邻接矩阵增强参数
        self.adj_enhancement = nn.Parameter(torch.tensor(2.0))  # 初始值大于1

    def forward(self, src, adj_matrix=None, src_key_padding_mask=None):
        """
        src: [batch_size, seq_len, d_model]
        adj_matrix: [batch_size, seq_len, seq_len] 或 [seq_len, seq_len]
        """
        batch_size, seq_len, d_model = src.shape

        # 如果有邻接矩阵，调整注意力计算
        if adj_matrix is not None:
            # 确保adj_matrix的维度正确
            if adj_matrix.dim() == 2:
                adj_matrix = adj_matrix.unsqueeze(0).expand(batch_size, -1, -1)

            # 增强有引用关系的位置
            # 这里使用加法，使得邻接矩阵的值大于1的位置（即引用关系）获得更高的注意力权重
            # 我们使用可学习的参数来调整增强的程度
            enhanced_adj = torch.where(
                adj_matrix > 1.0,
                adj_matrix * self.adj_enhancement,
                adj_matrix
            )

            # 将增强后的邻接矩阵转换为attention bias
            # 注意：PyTorch的attention期望mask中False的位置会被忽略
            # 我们使用加法偏置，正值会增加注意力权重
            attention_bias = enhanced_adj

            # 自注意力计算（加入偏置）
            src2, attn_weights = self.self_attn(
                src, src, src,
                attn_mask=attention_bias,
                key_padding_mask=src_key_padding_mask
            )
        else:
            # 普通的自注意力
            src2, attn_weights = self.self_attn(
                src, src, src,
                key_padding_mask=src_key_padding_mask
            )

        # 残差连接和归一化
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # 前馈网络
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class DMCModel(nn.Module):
    def __init__(self, num_senders=100, d_model=768):
        super().__init__()

        # 1. 文本内容编码器
        self.content_encoder = ContentEncoder(d_model=d_model)

        # 2. 其他属性编码器
        # 发送人编码
        self.sender_embedding = nn.Embedding(num_senders, d_model)

        # 时间编码（使用正弦位置编码的变体）
        self.time_encoding = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, d_model)
        )

        # 3. 属性聚合器（聚合内容、发送人、时间三个属性）
        self.property_aggregator = PropertyAggregator(
            num_properties=3,  # 内容、发送人、时间
            d_model=d_model
        )

        # 4. 消息间聚合编码器（带邻接矩阵干预）
        self.msg_interaction_layers = nn.ModuleList([
            ExtendedTransformerEncoderLayer(d_model=d_model) for _ in range(6)
        ])

        # 5. 回复头
        self.reply_header = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # 输出每个消息的回复得分
        )

        # 全局选择向量（用于计算与每个消息的相似度）
        self.global_selection_vector = nn.Parameter(torch.randn(1, 1, d_model))

        # 6. 内容生成解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=1024,
            batch_first=True,
            dropout=0.1
        )
        self.content_decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)

        # 输出投影层（到BERT词表）
        self.output_projection = nn.Linear(d_model, self.content_encoder.tokenizer.vocab_size)

        # 特殊token
        self.no_reply_token = nn.Parameter(torch.randn(1, d_model))

        # 用于内容生成的起始token embedding
        self.start_token_embedding = nn.Embedding(1, d_model)

        self.d_model = d_model

    def build_adjacency_matrix(self, msgs_info, device):
        """
        构建邻接矩阵
        规则：
        1. 基础矩阵全为1（表示所有消息之间的基本联系）
        2. 对角线设为0（不考虑自己到自己的特殊联系）
        3. 如果消息i引用了消息j，则X[i,j] = a（a>1且可训练）
        4. 行归一化，使得每行的和为1
        """
        seq_len = len(msgs_info)

        # 初始化基础邻接矩阵
        adj_matrix = torch.ones(seq_len, seq_len, device=device)

        # 对角线设为0
        adj_matrix.fill_diagonal_(0)

        # 处理引用回复关系
        for i, msg in enumerate(msgs_info):
            if msg['reply_to'] is not None:
                reply_to_id = msg['reply_to']
                # 找到被引用的消息在序列中的位置
                for j, other_msg in enumerate(msgs_info):
                    if other_msg.get('msg_id') == reply_to_id:
                        # 设置增强的权重，初始值为2.0，可训练
                        adj_matrix[i, j] = 2.0
                        break

        # 行归一化
        row_sums = adj_matrix.sum(dim=1, keepdim=True)
        adj_matrix = adj_matrix / row_sums.clamp(min=1e-8)

        return adj_matrix

    def encode_messages(self, msgs_info):
        """编码所有消息"""
        device = next(self.parameters()).device

        # 提取属性
        text_contents = [msg['content'] for msg in msgs_info]
        sender_ids = torch.tensor([msg['sender'] for msg in msgs_info], device=device)
        timestamps = torch.tensor([[msg['timestamp']] for msg in msgs_info],
                                  dtype=torch.float32, device=device)

        # 1. 编码文本内容（每个消息单独编码）
        content_embeddings = self.content_encoder(text_contents)  # [seq_len, d_model]

        # 2. 编码其他属性
        sender_embeddings = self.sender_embedding(sender_ids)  # [seq_len, d_model]
        time_embeddings = self.time_encoding(timestamps)  # [seq_len, d_model]

        # 3. 构建属性张量 [batch_size=1, seq_len, num_properties=3, d_model]
        properties = torch.stack([
            content_embeddings,  # 文本内容
            sender_embeddings,  # 发送人
            time_embeddings  # 时间
        ], dim=1).unsqueeze(0)  # [1, seq_len, 3, d_model]

        # 4. 聚合每个消息的属性
        aggregated_msgs = self.property_aggregator(properties)  # [1, seq_len, d_model]

        return aggregated_msgs

    def select_reply_target(self, msg_embeddings):
        """选择要回复的目标消息"""
        batch_size, seq_len, d_model = msg_embeddings.shape

        # 方法1: 使用全局选择向量计算相似度
        selection_vector = self.global_selection_vector.expand(batch_size, -1, -1)
        similarity_scores = torch.bmm(
            selection_vector,  # [batch_size, 1, d_model]
            msg_embeddings.transpose(1, 2)  # [batch_size, d_model, seq_len]
        ).squeeze(1)  # [batch_size, seq_len]

        # 方法2: 使用回复头计算得分（备用）
        reply_scores = self.reply_header(msg_embeddings).squeeze(-1)  # [batch_size, seq_len]

        # 结合两种方法
        combined_scores = similarity_scores + reply_scores

        # 应用softmax得到概率分布
        reply_probs = F.softmax(combined_scores, dim=-1)

        # 找到概率最大的消息
        max_prob, max_idx = torch.max(reply_probs, dim=-1)

        return reply_probs, max_idx, max_prob

    def forward(self, msgs_info, mode="train", target_tokens=None):
        """
        前向传播

        Args:
            msgs_info: 消息列表，每个元素是字典，包含：
                - 'content': 文本内容
                - 'sender': 发送人ID
                - 'timestamp': 时间戳
                - 'reply_to': 引用的消息ID（可为None）
                - 'msg_id': 消息ID
            mode: 'train' 或 'inference'
            target_tokens: 训练时的目标token序列
        """
        device = next(self.parameters()).device
        seq_len = len(msgs_info)

        # 1. 编码所有消息的属性并聚合
        msg_embeddings = self.encode_messages(msgs_info)  # [1, seq_len, d_model]

        # 2. 构建邻接矩阵
        adj_matrix = self.build_adjacency_matrix(msgs_info, device)  # [seq_len, seq_len]

        # 3. 消息间交互（使用邻接矩阵干预）
        encoder_output = msg_embeddings
        for layer in self.msg_interaction_layers:
            encoder_output = layer(encoder_output, adj_matrix=adj_matrix)

        # 4. 选择回复目标
        reply_probs, selected_idx, max_prob = self.select_reply_target(encoder_output)

        # 5. 决定是否回复特定消息
        threshold = 0.3  # 可调整的阈值
        if max_prob.item() < threshold:
            reply_target_idx = -1
            reply_target_emb = self.no_reply_token.unsqueeze(0).unsqueeze(0)  # [1, 1, d_model]
        else:
            reply_target_idx = selected_idx.item()
            reply_target_emb = encoder_output[:, selected_idx:selected_idx + 1, :]  # [1, 1, d_model]

        # 6. 内容生成
        if mode == "train" and target_tokens is not None:
            # 训练模式：使用teacher forcing

            # 创建解码器输入（右移一位）
            decoder_input = target_tokens[:, :-1]
            decoder_target = target_tokens[:, 1:]

            # 将token转换为embedding
            decoder_embeddings = self.content_encoder.bert.embeddings.word_embeddings(decoder_input)

            # 生成自注意力mask（防止看到未来信息）
            tgt_len = decoder_embeddings.shape[1]
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(device)

            # 解码
            decoder_output = self.content_decoder(
                tgt=decoder_embeddings,
                memory=encoder_output,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=None
            )

            # 输出投影
            logits = self.output_projection(decoder_output)  # [batch_size, tgt_len, vocab_size]

            return {
                'logits': logits,
                'reply_probs': reply_probs,
                'reply_target_idx': reply_target_idx,
                'decoder_target': decoder_target
            }

        else:
            # 推理模式：自回归生成
            max_gen_len = 50
            generated_tokens = []

            # 初始输入：开始token
            current_token = torch.tensor([[self.content_encoder.tokenizer.cls_token_id]],
                                         device=device)

            for step in range(max_gen_len):
                # 获取当前token的embedding
                current_embedding = self.content_encoder.bert.embeddings.word_embeddings(current_token)

                # 生成自注意力mask
                tgt_len = current_embedding.shape[1]
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(device)

                # 解码
                decoder_output = self.content_decoder(
                    tgt=current_embedding,
                    memory=encoder_output,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=None
                )

                # 获取最后一个token的输出
                last_output = decoder_output[:, -1:, :]  # [1, 1, d_model]

                # 融合回复目标信息
                if reply_target_idx != -1:
                    last_output = last_output + reply_target_emb

                # 输出投影
                logits = self.output_projection(last_output)  # [1, 1, vocab_size]

                # 采样（这里使用贪婪解码）
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

                # 检查是否生成结束token
                if next_token.item() == self.content_encoder.tokenizer.sep_token_id:
                    break

                generated_tokens.append(next_token.item())
                current_token = torch.cat([current_token, next_token], dim=1)

            # 将token ids转换为文本
            generated_text = self.content_encoder.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            )

            return {
                'generated_text': generated_text,
                'reply_probs': reply_probs,
                'reply_target_idx': reply_target_idx,
                'encoder_output': encoder_output
            }
