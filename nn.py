import torch
import numpy as np
import synth
import einops
from glotnet_wavenet.convolution_stack import ConvolutionStack

class DilatedConvStackStack( torch.nn.Module):
    def __init__(self, input_size, cond_size, hidden_size, output_size, dilations, causal, activation, kernel_size, use_skips = False, n_attention_heads=0) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dilations = dilations
        self.cond_size = cond_size
        self.causal = causal
        self.activation = activation
        self.use_skips = use_skips
        self.n_attention_heads = n_attention_heads

        input_size = (input_size + cond_size) if n_attention_heads > 0 else input_size


        self.conv1 = torch.nn.Conv1d(input_size,hidden_size, kernel_size=1, bias=True)
        self.dilated_stacks = torch.nn.ModuleList()

        self.attention_heads = n_attention_heads
        if self.n_attention_heads > 0:
            self.multihead_attentions = torch.nn.ModuleList()
            for dilation in dilations:
                self.multihead_attentions.append(torch.nn.MultiheadAttention(embed_dim=hidden_size, num_heads=self.n_attention_heads, batch_first=True)) 

        for dilation in dilations:
            self.dilated_stacks.append(ConvolutionStack(channels =hidden_size,
                                                         skip_channels =hidden_size,
                                                         dilations = dilation, 
                                                         kernel_size = kernel_size, 
                                                         cond_channels = cond_size if self.n_attention_heads == 0 else None, 
                                                         activation='relu'))
        self.conv2 = torch.nn.Conv1d(hidden_size, output_size, kernel_size=1, bias=True)

    def forward(self, x, cond):
        if self.n_attention_heads > 0:
            return self.forward_sa(x,cond)
        x = self.conv1(x)
        for stack in self.dilated_stacks:
            _, skips = stack(x, cond)
            if self.use_skips:
                x = torch.stack(skips, dim=0).sum(dim=0)
            else:
                x = _
        x = self.conv2(x)
        return x

    def forward_sa(self,x,cond):
        N_VOICES=6
        batch_x_channels, ft,time = x.shape
        batch = batch_x_channels // N_VOICES
        x = torch.cat([x,cond],dim=-2)
        x = self.conv1(x)
        for i,stack in enumerate(self.dilated_stacks):
            # apply layer norm
            x = einops.rearrange(x, '(b c) f t -> b c t f', b=batch, c=N_VOICES, t=time)
            x = torch.nn.functional.layer_norm(x, x.shape[1:])
            x = einops.rearrange(x, 'b c t f -> (b c) f t', b=batch, c=N_VOICES, t=time)

            x_attn = einops.rearrange(x, '(b c) f t -> (b t) c f', b=batch, c=N_VOICES, t=time)
            x_attn = self.multihead_attentions[i](x_attn, x_attn, x_attn)[0]
            x_attn = einops.rearrange(x_attn, '(b t) c f -> (b c) f t', b=batch, c=N_VOICES, t=time)
            x += x_attn
            out, skips = stack(x)
            if self.use_skips:
                out = torch.stack(skips, dim=0).sum(dim=0)
            x += out
        x = self.conv2(x)
        return x


class Fc(torch.nn.Module):
    def __init__(self, in_ch, out_ch, activation=torch.nn.LeakyReLU):
        super().__init__()
        self.fc = torch.nn.Linear(in_ch, out_ch)
        self.layer_norm = torch.nn.LayerNorm(out_ch)
        self.activation = activation()
    def forward(self, x):
        x = self.fc(x)
        x = self.layer_norm(x)
        x = self.activation(x)
        return x

class FcStack(torch.nn.Module):
    def __init__(self, in_ch, out_ch, layers=2, nonlinearity=torch.nn.LeakyReLU):
        super().__init__()
        first_layer = Fc(in_ch, out_ch, activation=nonlinearity)
        self.layers = torch.nn.Sequential(*[first_layer] + [Fc(out_ch, out_ch, activation=nonlinearity) for _ in range(layers-1)])
    def forward(self, x):
        return self.layers(x)

class RNN(torch.nn.Module):
    def __init__(self, ch, rnn_type='gru',bidirectional=True,return_sequences=True, n_layers=1):
        super().__init__()
        self.return_sequences = return_sequences
        if rnn_type == 'gru':
            self.rnn = torch.nn.GRU(ch, ch, batch_first=True, bidirectional=bidirectional, num_layers=n_layers)
        elif rnn_type == 'lstm':
            self.rnn = torch.nn.LSTM(ch, ch, batch_first=True, bidirectional=bidirectional, num_layers=n_layers)
        elif rnn_type == 'rnn':
            self.rnn = torch.nn.RNN(ch, ch, batch_first=True, bidirectional=bidirectional, num_layers=n_layers)

    def forward(self, x):
        if self.return_sequences:
            return self.rnn(x)[0]
        else:
            return self.rnn(x)[0][:,-1,:]






    