from __future__ import annotations

import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerDecoderLayer, TransformerDecoder
from transformers import AutoTokenizer, AutoModel

class BERTTextEncoder(nn.Module):
    def __init__(self, freeze_bert=False):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        self.bert = AutoModel.from_pretrained("bert-base-chinese")

        # 是否冻结BERT参数（如果不需要微调）
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, texts):
        """编码文本，返回CLS token的表示"""
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        # 将输入移动到模型所在的设备
        device = next(self.bert.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = self.bert(**inputs)

        # 使用CLS token的输出作为句子表示
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch_size, 768]

        return cls_embeddings

    def get_token_embeddings(self, input_ids):
        """获取token的embedding（用于解码器）"""
        return self.bert.embeddings.word_embeddings(input_ids)

    @property
    def vocab_size(self):
        return len(self.tokenizer)

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

class ExtendedTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=768, nhead=8, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        # 标准的TransformerEncoderLayer
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

        # 可学习的邻接矩阵参数
        self.learnable_adj = nn.Parameter(torch.randn(1, 1, 1))

    def forward(self, src, adj_matrix=None, src_mask=None):
        """
        src: [batch_size, seq_len, d_model]
        adj_matrix: [batch_size, seq_len, seq_len] 或 [seq_len, seq_len]
        """
        batch_size, seq_len, _ = src.shape

        # 如果有邻接矩阵，调整注意力计算
        if adj_matrix is not None:
            if adj_matrix.dim() == 2:
                adj_matrix = adj_matrix.unsqueeze(0).expand(batch_size, -1, -1)

            # 将邻接矩阵作为attention mask的偏置
            # 注意：这里使用加法，使得有引用的位置更容易被关注
            if src_mask is None:
                src_mask = torch.zeros_like(adj_matrix)

            # 结合可学习的参数
            learnable_bias = self.learnable_adj.expand(batch_size, seq_len, seq_len)
            combined_mask = src_mask + adj_matrix * learnable_bias

            # 转换为attention mask（注意：PyTorch的attention期望mask的布尔形式）
            attention_mask = combined_mask > 0
        else:
            attention_mask = src_mask

        # 自注意力层
        src2, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=attention_mask,
            need_weights=True
        )

        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # 前馈网络
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class PropertyAggregator(nn.Module):
    """聚合单个消息的三个属性"""

    def __init__(self, d_model=768, nhead=8, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=1024,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 用于将三个属性的聚合表示融合为一个
        self.fusion = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model)
        )

class DMCModel(nn.Module):
    def __init__(self, num_senders=100, d_model=768):
        super().__init__()
        # 文本编码器
        self.text_encoder = BERTTextEncoder(freeze_bert=True)

        # 属性编码器
        # 发送人编码
        self.sender_embedding = nn.Embedding(num_senders, d_model)

        # 时间编码（使用位置编码类似的方法）
        self.time_encoding = nn.Linear(1, d_model)

        # 引用回复编码（如果有引用的消息，编码其ID）
        self.reply_embedding = nn.Embedding(1000, d_model)  # 假设最多1000条消息

        # 属性聚合器（每个消息的三个属性）
        self.property_aggregator = PropertyAggregator(d_model=d_model)

        # 消息间聚合编码器（带邻接矩阵）
        self.msg_encoder_layers = nn.ModuleList([
            ExtendedTransformerEncoderLayer(d_model=d_model) for _ in range(6)
        ])

        # 回复头
        self.reply_header = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

        # 选择向量（用于计算与每个消息的相似度）
        self.selection_vector = nn.Parameter(torch.randn(1, 1, d_model))

        # 内容生成解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=1024,
            batch_first=True
        )
        self.content_decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)

        # 输出投影层（到词表）
        self.output_projection = nn.Linear(d_model, self.text_encoder.vocab_size)

        # 可训练参数a（用于邻接矩阵中引用关系的权重）
        self.reply_weight = nn.Parameter(torch.tensor(2.0))

        # 用于无回复情况下的特殊token
        self.no_reply_token = nn.Parameter(torch.randn(1, d_model))

        self.d_model = d_model

    def build_adjacency_matrix(self, msgs_info, device):
        """
        构建邻接矩阵
        msgs_info: 消息列表，每个元素包含'reply_to'字段（消息ID或None）
        返回: [seq_len, seq_len]的邻接矩阵
        """
        seq_len = len(msgs_info)

        # 初始化全1矩阵（基础联系）
        adj_matrix = torch.ones(seq_len, seq_len, device=device)

        # 对角线设为0（不考虑自己到自己的特殊联系）
        adj_matrix.fill_diagonal_(0)

        # 处理引用回复关系
        for i, msg in enumerate(msgs_info):
            if msg['reply_to'] is not None:
                reply_to_id = msg['reply_to']
                # 找到被引用的消息在序列中的位置
                # 注意：这里假设被引用的消息一定在序列中
                for j, other_msg in enumerate(msgs_info):
                    if other_msg.get('msg_id') == reply_to_id:
                        # 设置引用关系的权重（可训练参数a）
                        adj_matrix[i, j] = self.reply_weight
                        break

        # 行归一化（使得每行的和为1）
        row_sums = adj_matrix.sum(dim=1, keepdim=True)
        adj_matrix = adj_matrix / row_sums

        return adj_matrix

    def forward(self, msgs_info, mode="train", target_tokens=None):
        """
        msgs_info: 消息列表，每个元素是包含以下字段的字典:
            - 'content': 文本内容
            - 'sender': 发送人ID
            - 'timestamp': 时间戳（浮点数）
            - 'reply_to': 引用的消息ID（可为None）
            - 'msg_id': 消息在序列中的ID
        """
        batch_size = 1  # 假设批量大小为1
        seq_len = len(msgs_info)
        device = next(self.parameters()).device

        # 1. 编码每个消息的三个属性
        text_contents = [msg['content'] for msg in msgs_info]
        sender_ids = torch.tensor([msg['sender'] for msg in msgs_info], device=device)
        timestamps = torch.tensor([[msg['timestamp']] for msg in msgs_info],
                                  dtype=torch.float32, device=device)

        # 编码文本内容
        with torch.no_grad():  # 因为BERT被冻结
            text_embeddings = self.text_encoder(text_contents)  # [seq_len, d_model]

        # 编码发送人
        sender_embeddings = self.sender_embedding(sender_ids)  # [seq_len, d_model]

        # 编码时间
        time_embeddings = self.time_encoding(timestamps)  # [seq_len, d_model]

        # 构建属性张量 [batch_size, seq_len, num_properties=3, d_model]
        properties = torch.stack([text_embeddings, sender_embeddings, time_embeddings],
                                 dim=1).unsqueeze(0)  # [1, seq_len, 3, d_model]

        # 2. 聚合每个消息的三个属性
        aggregated_msgs = self.property_aggregator(properties)  # [1, seq_len, d_model]

        # 3. 构建邻接矩阵
        adj_matrix = self.build_adjacency_matrix(msgs_info, device)  # [seq_len, seq_len]

        # 4. 通过扩展的Transformer层进行消息间聚合
        encoder_output = aggregated_msgs
        for layer in self.msg_encoder_layers:
            encoder_output = layer(encoder_output, adj_matrix=adj_matrix)

        # encoder_output: [1, seq_len, d_model]

        # 5. 回复头：决定是否回复以及回复哪条消息
        # 方法1: 使用选择向量计算相似度
        selection_vector = self.selection_vector.expand(batch_size, -1, -1)  # [1, 1, d_model]

        # 计算相似度得分
        similarity_scores = torch.bmm(
            selection_vector,  # [1, 1, d_model]
            encoder_output.transpose(1, 2)  # [1, d_model, seq_len]
        ).squeeze(1)  # [1, seq_len]

        # 应用softmax得到概率
        reply_probs = torch.softmax(similarity_scores, dim=-1)  # [1, seq_len]

        # 找到概率最大的消息
        max_prob, max_idx = torch.max(reply_probs, dim=-1)

        # 设置阈值：如果最大概率低于阈值，则不回复特定消息
        threshold = 0.3
        if max_prob.item() < threshold:
            reply_target_idx = -1
            # 使用特殊token表示无回复
            reply_target_emb = self.no_reply_token.expand(batch_size, -1, -1)  # [1, 1, d_model]
        else:
            reply_target_idx = max_idx.item()
            # 获取目标消息的表示
            reply_target_emb = encoder_output[:, reply_target_idx:reply_target_idx + 1, :]  # [1, 1, d_model]

        # 6. 内容生成（文本解码）
        if mode == "train" and target_tokens is not None:
            # 训练模式：使用teacher forcing
            target_embeddings = self.text_encoder.get_token_embeddings(target_tokens)
            target_embeddings = target_embeddings.unsqueeze(0)  # [1, target_len, d_model]

            # 生成自注意力mask（防止看到未来信息）
            tgt_len = target_embeddings.shape[1]
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(device)

            # 解码
            decoder_output = self.content_decoder(
                tgt=target_embeddings,
                memory=encoder_output,
                tgt_mask=tgt_mask,
                tgt_is_causal=True
            )

            # 输出投影
            logits = self.output_projection(decoder_output)  # [1, target_len, vocab_size]

            return {
                'logits': logits,
                'reply_probs': reply_probs,
                'reply_target_idx': reply_target_idx
            }

        else:
            # 推理模式：自回归生成
            max_gen_len = 50
            generated_tokens = []

            # 初始输入：开始token（使用CLS token或特殊token）
            current_token = torch.tensor([[self.text_encoder.tokenizer.cls_token_id]],
                                         device=device)

            for step in range(max_gen_len):
                # 获取当前token的embedding
                current_embedding = self.text_encoder.get_token_embeddings(current_token)

                # 生成自注意力mask
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                    current_embedding.shape[1]
                ).to(device)

                # 解码
                decoder_output = self.content_decoder(
                    tgt=current_embedding,
                    memory=encoder_output,
                    tgt_mask=tgt_mask,
                    tgt_is_causal=True
                )

                # 获取最后一个token的输出
                last_output = decoder_output[:, -1:, :]  # [1, 1, d_model]

                # 输出投影
                logits = self.output_projection(last_output)  # [1, 1, vocab_size]

                # 采样（这里使用贪婪解码）
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

                # 检查是否生成结束token
                if next_token.item() == self.text_encoder.tokenizer.sep_token_id:
                    break

                generated_tokens.append(next_token.item())
                current_token = torch.cat([current_token, next_token], dim=1)

            # 将token ids转换为文本
            generated_text = self.text_encoder.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            )

            return {
                'generated_text': generated_text,
                'reply_probs': reply_probs,
                'reply_target_idx': reply_target_idx,
                'encoder_output': encoder_output
            }
