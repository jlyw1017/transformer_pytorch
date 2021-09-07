import torch
from model.common_layers import (positional_encoding,
                                 point_wise_feed_forward_network,
                                 MultiHeadAttention)

class EncoderLayer(torch.nn.Module):
    def __init__(self, 
                 d_model,
                 num_heads,
                 d_feedforward,
                 dropout_rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads, 64, 64)
        self.ffn = point_wise_feed_forward_network(d_model, d_feedforward)

        self.layernorm1 = torch.nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.layernorm2 = torch.nn.LayerNorm(normalized_shape=d_model, eps=1e-6)

        self.dropout1 = torch.nn.Dropout(dropout_rate)
        self.dropout2 = torch.nn.Dropout(dropout_rate)

    def forward(self, x, mask):
        # attention_output, _ = self.mha(x, x, x, mask)
        # attention_output = self.dropout1(attention_output)
        # out1 = self.layernorm1(x + attention_output)

        # ffn_output = self.ffn(out1)
        # ffn_output = self.dropout2(ffn_output)
        # out2 = self.layernorm2(out1 + ffn_output)

        # return out2
        attn_output, _ = self.mha(x, mask)  # =>[b, seq_len, d_model]
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)  # 残差&层归一化 =>[b, seq_len, d_model]

        ffn_output = self.ffn(out1)  # =>[b, seq_len, d_model]
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)  # 残差&层归一化 =>[b, seq_len, d_model]

        return out2  # [b, seq_len, d_model]

# end class EncoderLayer

class Encoder(torch.nn.Module):
    """Encodes the inputs.

    Attributes:
        vocab_dict_size (int): size of the vocabulary dict size.
        padded_sentence_length (int): max sentence length.
        num_encoder_layers (int): layer num of encoder, each layer consists of a
    multiheadattention layer and a residual layer.
        d_model (int): embeds the input word into a vector as
    d_model dimension and it is the same as row number of wq, wk and wv.


    """
    def __init__(self,
                 vocab_dict_size,
                 padded_sentence_length,
                 num_encoder_layers,
                 d_model,
                 num_heads,
                 dff,  # 点式前馈网络内层fn的维度
                 rate=0.1):
        super(Encoder, self).__init__()

        self.num_layers = num_encoder_layers
        self.d_model = d_model

        self.embedding = torch.nn.Embedding(num_embeddings=vocab_dict_size,
                                            embedding_dim=d_model)
        self.pos_encoding = positional_encoding(padded_sentence_length,
                                                d_model)
        # =>[1, max_pos_encoding, d_model=512]

        # self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate).cuda() for _ in range(num_layers)] # 不行
        self.enc_layers = torch.nn.ModuleList(
            [EncoderLayer(d_model, num_heads, dff, rate)
             for _ in range(num_encoder_layers)])

        self.dropout = torch.nn.Dropout(rate)

    # x [b, inp_seq_len]
    # mask [b, 1, 1, inp_sel_len]
    def forward(self, x, mask):
        inp_seq_len = x.shape[-1]

        # adding embedding and position encoding
        x = self.embedding(x)  # [b, inp_seq_len]=>[b, inp_seq_len, d_model]
        # 缩放 embedding 原始论文的3.4节有提到： In the embedding layers, we multiply those weights by \sqrt{d_model}.
        x *= torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        pos_encoding = self.pos_encoding[:, :inp_seq_len, :]
        pos_encoding = pos_encoding.to("cuda:0")  # ###############
        x += pos_encoding  # [b, inp_seq_len, d_model]

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask)  # [b, inp_seq_len, d_model]=>[b, inp_seq_len, d_model]
        return x  # [b, inp_seq_len, d_model]
# end class Encoder