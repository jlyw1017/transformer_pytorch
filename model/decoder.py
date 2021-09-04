import torch
from torch.nn.modules import dropout
from model.utils import (point_wise_feed_forward_network,
                         positional_encoding,
                         MultiHeadAttention)


class DecoderLayer(torch.nn.Module):
    def __init__(self,
                 d_model,
                 num_heads,
                 d_feedforward,
                 dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads, 64, 64)
        self.mha2 = MultiHeadAttention(d_model, num_heads, 64, 64)

        self.ffn = point_wise_feed_forward_network(d_model, d_feedforward)

        self.layernorm1 = torch.nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.layernorm2 = torch.nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.layernorm3 = torch.nn.LayerNorm(normalized_shape=d_model, eps=1e-6)

        self.dropout1 = torch.nn.Dropout(dropout_rate)
        self.dropout2 = torch.nn.Dropout(dropout_rate)
        self.dropout3 = torch.nn.Dropout(dropout_rate)

    def forward(self, x, enc_output, look_ahead_mask, padding_mask):

        attn1, attn_weights_block1 = self.mha1(x, look_ahead_mask)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(out1, enc_output, enc_output, padding_mask)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2

class Decoder(torch.nn.Module):
    def __init__(self,
                 num_layeres,
                 d_model,
                 num_heads,
                 d_feedforward,
                 target_vocab_size,
                 maximum_position_encoding,
                 dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layeres

        self.embedding = torch.nn.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = torch.nn.ModuleList(
                [DecoderLayer(d_model, num_heads, d_feedforward) for _ in range(num_layeres)]
            )
        
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        targ_seq_len = x.shape[-1]

        attention_weights = {}

        # adding embedding and position encoding
        x = self.embedding(x)  # [b, targ_seq_len]=>[b, targ_seq_len, d_model]
        # 缩放 embedding 原始论文的3.4节有提到： In the embedding layers, we multiply those weights by \sqrt{d_model}.
        x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        # x += self.pos_encoding[:, :targ_seq_len, :]  # [b, targ_seq_len, d_model]
        pos_encoding = self.pos_encoding[:, :targ_seq_len, :]  # [b, targ_seq_len, d_model]
        pos_encoding = pos_encoding.cuda("cuda:0") # ###############
        x += pos_encoding  # [b, inp_seq_len, d_model]

        x = self.dropout(x)

        for i in range(self.num_layers):
            x, attn_block1, attn_block2 = self.dec_layers[i](x, enc_output, look_ahead_mask, padding_mask)
            # => [b, targ_seq_len, d_model], [b, num_heads, targ_seq_len, targ_seq_len], [b, num_heads, targ_seq_len, inp_seq_len]

            attention_weights[f'decoder_layer{i + 1}_block1'] = attn_block1
            attention_weights[f'decoder_layer{i + 1}_block2'] = attn_block2

        return x, attention_weights