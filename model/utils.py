from numpy.lib.shape_base import expand_dims
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.modules.activation import ReLU


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, 2*(i//2) / np.float(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):  # d_model是位置编码的长度，相当于position encoding的embedding_dim？
    """Encodes positional information to """
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],  # [50, 1]
                            np.arange(d_model)[np.newaxis, :],  # [1, d_model=512]
                            d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])  # 2i
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])  # 2i+2

    pos_encoding = angle_rads[np.newaxis, ...]  # [50,512]=>[1,50,512]
    return torch.tensor(pos_encoding, dtype=torch.float32)


def create_padding_mask(seq, pad):
    seq = torch.eq(seq, torch.tensor(pad)).float()
    return seq[:, np.newaxis, np.newaxis, :]


def create_look_ahead_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask


def scaled_dot_product_attention(q, k, v, mask=None):
    """计算注意力权重。
    q, k, v 必须具有匹配的前置维度。
    k, v 必须有匹配的倒数第二个维度，例如：seq_len_k = seq_len_v。
    虽然 mask 根据其类型（填充或前瞻）有不同的形状，
    但是 mask 必须能进行广播转换以便求和。

    参数:
        q: 请求的形状 == (..., seq_len_q, depth)
        k: 主键的形状 == (..., seq_len_k, depth)
        v: 数值的形状 == (..., seq_len_v, depth_v)
        mask: Float 张量，其形状能转换成
            (..., seq_len_q, seq_len_k)。默认为None。

    返回值:
        输出，注意力权重
    """

    matmul_qk = torch.matmul(q, k.transpose(-2, -1))
    depth_k = torch.tensor(k.shape[-1], dtype=torch.float32)
    scaled_attion_logits = matmul_qk / torch.sqrt(depth_k)

    if mask is not None:
        scaled_attion_logits += (mask * -1e9)

    attention_weights = torch.nn.functional.softmax(scaled_attion_logits, dim=-1)

    output = torch.matmul(attention_weights, v)

    return output, attention_weights


def point_wise_feed_forward_network(d_model, d_feedforward):
    feed_forward_net = torch.nn.Sequential(
        torch.nn.Linear(d_model, d_feedforward),
        torch.nn.ReLU(),
        torch.nn.Linear(d_feedforward, d_model)
    )

    return feed_forward_net


class MultiHeadAttention(torch.nn.Module):
    """The definition MultiHeadAttention Layer.

    input: [batch, padded_sentence_length, d_model]
    Creates the q and k with matrix wq and wk as [d_model, dq] dq = dk must
    be satisfied.
    dq doesn't has to be the same as d_model, in the paper it is 64.

    softmax(q * kt/ sqrt(dk)): [sentence_length, sentence_length]

    Creates v with matrix wv [d_model, dv] get v with [sentence_length, dv]
    single head z = softmax(q * kt/ sqrt(dk)) * v with [sentence_length, dv]

    wq: [num_head * dv, d_model] the column of wo must be d_model to keep the
    feature size of each word.

    In the parpar "All you need is attention" dq = dk = dv = d_model / num head.
    Actually, dq = dk,  dz = dv
    The num_head is not related to d_model.
    """

    def __init__(self, d_model, num_heads, dqk, dv):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dqk = dqk
        self.dv = dv

        #assert d_model % self.num_heads == 0, "d_model must be divisible by
        # num_heads!"
        #self.depth = d_model // self.num_heads

        self.wq = torch.nn.Linear(d_model, dqk)
        self.wk = torch.nn.Linear(d_model, dqk)
        self.wv = torch.nn.Linear(d_model, dv)
        self.woutput = torch.nn.Linear(num_heads * dv, d_model)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.transpose(1,2)

    def forward(self, x, mask):
        batch_size = x.shape[0]

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = scaled_attention.transpose(1, 2)
        concat_attention = scaled_attention.reshape(batch_size, -1, self.d_model)

        output = self.woutput(concat_attention)
        return output, attention_weights
# end class MultiheadAttention
