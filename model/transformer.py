import torch
from model.encoder import Encoder
from model.decoder import Decoder

class Transformer(torch.nn.Module):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 d_feedforward,
                 input_vocab_size,
                 target_vocab_size,
                 pe_input,  # input max_pos_encoding
                 pe_target,  # input max_pos_encoding
                 dropout_rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, 
                               d_model, 
                               num_heads, 
                               d_feedforward, 
                               input_vocab_size,
                               pe_input, 
                               dropout_rate)

        self.decoder = Decoder(num_layers, 
                               d_model, 
                               num_heads,
                               d_feedforward, 
                               target_vocab_size,
                               pe_target, 
                               dropout_rate)

        self.final_layer = torch.nn.Linear(d_model, target_vocab_size)

    def forward(self, x, target, enc_padding_mask, 
                look_ahead_mask, dec_padding_mask):

        enc_output = self.encoder(x, enc_padding_mask)
        dec_output, attention_weights = self.decoder(target, enc_output, 
                                                     look_ahead_mask, dec_padding_mask)
        
        final_output = self.final_layer(dec_output)

        return final_output, attention_weights