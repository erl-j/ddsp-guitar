import math

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn

from glotnet_wavenet.convolution_stack import ConvolutionStack
from synthesis_model import DilatedConvStackStack


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))

class SACNNStack(torch.nn.Module):
    def __init__(self,
        input_size,
        output_size,
        hidden_size,
        n_channels,
        n_blocks,
        n_heads,
        dilations,
        kernel_size,
        activation,
        norm_type,
    ):
        super(SACNNStack, self).__init__()
        self.n_channels = n_channels
        self.input_size = input_size
        self.output_size = output_size
        self.n_blocks = n_blocks
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.dilations = dilations
        self.kernel_size = kernel_size
        self.activation = activation
        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size,hidden_size),
            torch.nn.ReLU()
        )
        # create N sarn blocks 
        self.model =nn.Sequential(
            *[SACNNBlock(input_channels=n_channels,
                        hidden_size=hidden_size,
                        num_heads=n_heads,
                        dilations=dilations,
                        kernel_size=kernel_size,
                        activation=activation,
                        norm_type=norm_type)
            for _ in range(self.n_blocks)])

        self.output_layer = torch.nn.Sequential(torch.nn.Linear(hidden_size, output_size))

    def forward(self, x, cond=None):
        if cond is not None:
            x = torch.cat([x, cond], dim=-1)
        x = self.input_layer(x)
        x = self.model(x)
        x = self.output_layer(x)
        return x

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        pe = einops.rearrange(self.pe[:x.size(2)], 't b e ->  b 1 t e')
        x = x + pe
        return x

class FullTransformerModel(torch.nn.Module):
    def __init__(self, n_layers, n_heads, hidden_size):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.positional_encoding = PositionalEncoding(hidden_size, max_len=1000)
        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=n_heads,
                dim_feedforward=hidden_size,
                # dropout=0.1,
                activation='relu',
                norm_first=True,
                batch_first=True
            ),
            num_layers=n_layers,

        )

        self.decoder = torch.nn.TransformerDecoder(
            torch.nn.TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=n_heads,
                dim_feedforward=hidden_size,
                norm_first=True,
                batch_first=True
            )
            ,
                num_layers=n_layers,

        )


        # create padding token weight
        self.padding_token = nn.Parameter(torch.randn(1, 1, hidden_size))

    def forward(self, x):
        batch, channel, time, feature = x.shape

        x = self.positional_encoding(x)
        x = einops.rearrange(x, 'b c t f -> b (c t) f')

        causal_mask = torch.triu(torch.ones(channel*time,channel*time, device=x.device), diagonal=1).bool()

        memory = self.encoder(x)
        expanded_padding_token = self.padding_token.expand(batch, 1, self.hidden_size)
        x_shifted = torch.cat([expanded_padding_token, x[:, :-1, :]], dim=1)
        out = self.decoder(x_shifted, memory, tgt_mask=causal_mask)
        out = einops.rearrange(out, 'b (c t) f -> b c t f', c=channel, t=time)
        return out

class TransformerEncoderModel(torch.nn.Module):
    def __init__(self, n_layers, n_heads, hidden_size):
        super(TransformerEncoderModel, self).__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hidden_size = hidden_size

        self.positional_encoding = PositionalEncoding(self.hidden_size, max_len=1000)

        self.transformer_encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=self.hidden_size,   
                nhead=self.n_heads,
                dim_feedforward=self.hidden_size,
                dropout=0.1,
                activation='relu',
                norm_first= True,
                batch_first=True
            ),
            num_layers=self.n_layers
        )
        self.output_layer = torch.nn.Sequential(torch.nn.Linear(self.hidden_size, 4))
    
    def forward(self, x):
        batch, channel, time, feature = x.shape
        x = self.positional_encoding(x)
        x = einops.rearrange(x, 'b c t f -> b (c t) f')
        out = self.transformer_encoder(x)
        out = einops.rearrange(out, 'b (c t) f -> b c t f', c=channel, t=time)
        return out
    
class SACNNBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        num_heads: int,
        hidden_size: int,
        dilations: list,
        kernel_size: int,
        activation: str,
        norm_type: str
    ):
        super(SACNNBlock, self).__init__()
        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dilations = dilations
        self.kernel_size = kernel_size
        self.activation = activation

        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True)
        self.time_model = ConvolutionStack(
            channels = hidden_size,
            skip_channels=hidden_size,
            kernel_size=kernel_size,
            dilations=dilations,
            activation=activation,
            causal=False,
            cond_channels=None,
        )

        self.norm_type = norm_type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channel, input_time, input_features = x.shape
        assert input_features == self.hidden_size, f"Input features {self.hidden_size} does not match expected {self.hidden_size}"
        assert channel == self.input_channels, f"Input channels {channel} does not match expected {self.input_channels}"

        if self.norm_type is not None:
            if self.norm_type == 'layer norm c t f':
                norm_shape = x.shape[1:]       
                x = F.layer_norm(x, norm_shape)
            else:
                raise ValueError(f"Invalid norm type '{self.norm_type}'")
        
        # x: [batch, channel, time, feature]
        batch, channel, input_time, input_features = x.shape

        x_attn = einops.rearrange(x, 'b c t f -> (b t) c f')
        x_attn = self.self_attn(x_attn, x_attn, x_attn)[0]
        x_attn = einops.rearrange(x_attn, '(b t) c f -> b c t f', b=batch, t=input_time, c=channel, f=input_features)
        # x_attn: [batch, channel, time, feature]

        x = x + x_attn  # Add residual connection after self-attention

        x_rnn, _ = self.time_model(
            einops.rearrange(x, 'b c t f -> (b c) f t',  b=batch, t=input_time, c=channel, f=input_features)
            )
        x_rnn = einops.rearrange(x_rnn, '(b c) f t -> b c t f', b=batch, t=input_time, c=channel, f=input_features)
        # x_rnn: [batch, channel, time, hidden_size]

        output = x + x_rnn  # Add residual connection after RNN

        return output
    

class SARNNBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        num_heads: int,
        hidden_size: int,
        rnn_type: str,
        n_rnn_layers_per_block: int,
        norm_type: str
    ):
        super(SARNNBlock, self).__init__()
        self.input_channels = input_channels
        self.hidden_size = hidden_size
        self.num_layers = n_rnn_layers_per_block
        self.rnn_type = rnn_type

        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, batch_first=True)
        self.rnn = self._create_rnn(rnn_type, hidden_size, n_rnn_layers_per_block)

        self.norm_type = norm_type

    def _create_rnn(self, rnn_type: str, hidden_size: int, num_layers: int) -> nn.Module:
        if rnn_type.lower() == 'rnn':
            return nn.RNN(hidden_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        elif rnn_type.lower() == 'lstm':
            return nn.LSTM(hidden_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        elif rnn_type.lower() == 'gru':
            return nn.GRU(hidden_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        else:
            raise ValueError(f"Invalid RNN type '{rnn_type}', supported types are 'RNN', 'LSTM', and 'GRU'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channel, input_time, input_features = x.shape
        assert input_features == self.hidden_size, f"Input features {self.hidden_size} does not match expected {self.hidden_size}"
        assert channel == self.input_channels, f"Input channels {channel} does not match expected {self.input_channels}"

        if self.norm_type is not None:
            if self.norm_type == 'layer norm c t f':
                norm_shape = x.shape[1:]       
                x = F.layer_norm(x, norm_shape)
            else:
                raise ValueError(f"Invalid norm type '{self.norm_type}'")
        
        # x: [batch, channel, time, feature]

        x_attn = einops.rearrange(x, 'b c t f -> (b t) c f')
        x_attn = self.self_attn(x_attn, x_attn, x_attn)[0]
        x_attn = einops.rearrange(x_attn, '(b t) c f -> b c t f', b=x.shape[0], c=self.input_channels, t=input_time)
        # x_attn: [batch, channel, time, feature]

        x = x + x_attn  # Add residual connection after self-attention

        x_rnn, _ = self.rnn(einops.rearrange(x, 'b c t f -> (b c) t f'))
        # sum hidden states from forward and backward RNNs
        x_rnn = x_rnn[:, :, :self.hidden_size] + x_rnn[:, :, self.hidden_size:]
        x_rnn = einops.rearrange(x_rnn, '(b c) t f -> b c t f', b=x.shape[0], c=self.input_channels, t=input_time)
        # x_rnn: [batch, channel, time, hidden_size]

        output = x + x_rnn  # Add residual connection after RNN

        return output

class IndependentStringStackModel(torch.nn.Module):
    def __init__(self, string_embedding_size):
        super(IndependentStringStackModel, self).__init__()

        self.string_embedding_size=string_embedding_size

        self.string_embedding_layer = torch.nn.Embedding(6, string_embedding_size)

        self.hidden_size = (string_embedding_size + 2) 

        self.model = DilatedConvStackStack(
                                        input_size = self.hidden_size,
                                        hidden_size=self.hidden_size,
                                        cond_size=None,
                                        output_size=self.hidden_size,
                                        dilations=[[1,2,4,8,16,32,64,128]]*4,
                                        causal=False,
                                        activation="gated",
                                        kernel_size=3)

        self.output_layer = torch.nn.Sequential(torch.nn.Linear(self.hidden_size, 4))

    def forward(self, inputs):
        midi_pitch = inputs["midi_pitch_scaled"]
        device = midi_pitch.device
        midi_velocity = inputs["midi_pseudo_velocity"]-1.0
        batch, channel, t, _ = midi_pitch.shape

        string_index = torch.arange(0, channel, device=device)
        string_z = self.string_embedding_layer(string_index)
        string_z  = einops.repeat(string_z, 'c ft-> b c t ft', b=batch, t=t, c=channel)

        z = torch.cat([midi_pitch, midi_velocity, string_z], dim=-1)

        z  = einops.rearrange(z, 'b c t ft -> (b c) ft t', c=channel, t=t, b=batch)

        out = self.model(z,None)

        out = einops.rearrange(out, '(b c) ft t -> b c t ft', c=channel, t=t, b=batch)

        out = self.output_layer(out)

        output={}

        output["hex_f0_scaled_hat"] = torch.sigmoid(out[...,0].unsqueeze(-1))
        output["hex_loudness_scaled_hat"] = torch.nn.functional.softplus(out[...,1].unsqueeze(-1))
        output["hex_periodicity_hat"] = torch.sigmoid(out[...,2].unsqueeze(-1))
        output["hex_centroid_scaled_hat"] = torch.sigmoid(out[...,3].unsqueeze(-1))
        return output


class MergedDilatedConvStackModel(torch.nn.Module):
    def __init__(self, string_embedding_size):
        super( MergedDilatedConvStackModel, self).__init__()

        self.string_embedding_size=string_embedding_size

        self.string_embedding_layer = torch.nn.Embedding(6, string_embedding_size)

        self.hidden_size = (string_embedding_size + 2) * 6 

        self.model = DilatedConvStackStack(
                                        input_size = self.hidden_size,
                                        hidden_size=self.hidden_size,
                                        cond_size=None,
                                        output_size=self.hidden_size,
                                        dilations=[[1,2,4,8,16,32,64,128]]*4,
                                        causal=False,
                                        activation="gated",
                                        kernel_size=3,
                                        use_skips=False)

        self.output_layer = torch.nn.Sequential(torch.nn.Linear(self.hidden_size // 6, 4*6))

    def forward(self, inputs):
        midi_pitch = inputs["midi_pitch_scaled"]
        device = midi_pitch.device
        midi_velocity = inputs["midi_pseudo_velocity"]-1.0
        batch, channel, t, _ = midi_pitch.shape

        string_index = torch.arange(0, channel, device=device)[:,None]
        string_z = self.string_embedding_layer(string_index)
        string_z  = einops.repeat(string_z, 'c 1 ft-> b c t ft', b=batch, t=t)

        z = torch.cat([midi_pitch, midi_velocity, string_z], dim=-1)

        z  = einops.rearrange(z, 'b c t ft -> b (c ft) t')

        out = self.model(z,None)

        out = einops.rearrange(out, 'b (c ft) t -> b c t ft', c=channel, t=t)

        out = self.output_layer(out)

        output={}

        output["hex_f0_scaled_hat"] = torch.sigmoid(out[...,0].unsqueeze(-1))
        output["hex_loudness_scaled_hat"] = torch.nn.functional.softplus(out[...,1].unsqueeze(-1))
        output["hex_periodicity_hat"] = torch.sigmoid(out[...,2].unsqueeze(-1))
        output["hex_centroid_scaled_hat"] = torch.sigmoid(out[...,3].unsqueeze(-1))
        return output

class MergedRepresentationTransformerModel(torch.nn.Module):
    def __init__(self, n_layers, n_heads, string_embedding_size):
        super(MergedRepresentationTransformerModel, self).__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads

        self.string_embedding_size=string_embedding_size

        self.string_embedding_layer = torch.nn.Embedding(6, string_embedding_size)

        self.hidden_size = (string_embedding_size + 2) * 6 + 1

        self.transformer_encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=self.hidden_size,   
                nhead=self.n_heads,
                dim_feedforward=self.hidden_size,
                dropout=0.1,
                activation='relu',
                norm_first= True,
                batch_first=True
            ),
            num_layers=self.n_layers
        )

        self.output_layer = torch.nn.Sequential(torch.nn.Linear(self.hidden_size, 4*6), torch.nn.Sigmoid())
    
    def forward(self, inputs):
        midi_pitch = inputs["midi_pitch_scaled"]
        device = midi_pitch.device
        midi_velocity = inputs["midi_pseudo_velocity"]
        batch, channel, t, _ = midi_pitch.shape

        string_index = torch.arange(0, channel, device=device)[:,None]
        string_z = self.string_embedding_layer(string_index)
        string_z  = einops.repeat(string_z, 'c 1 ft-> b c t ft', b=batch, t=t)

        z = torch.cat([midi_pitch, midi_velocity, string_z], dim=-1)

        z  = einops.rearrange(z, 'b c t ft -> b t (c ft)')

        time_z = einops.repeat(torch.arange(0, t, device=device), 't -> b t 1', b=batch)
        
        z = torch.cat([z, time_z], dim=-1)
        
        out = self.transformer_encoder(z)
        out = self.output_layer(out)

        out = einops.rearrange(out, 'b t (c ft) -> b c t ft', c=channel, t=t)

        output={}

        output["hex_f0_scaled_hat"] = out[...,0].unsqueeze(-1)
        output["hex_loudness_scaled_hat"] = out[...,1].unsqueeze(-1)
        output["hex_periodicity_hat"] = out[...,2].unsqueeze(-1)
        output["hex_centroid_scaled_hat"] = out[...,3].unsqueeze(-1)
        return output
    
class FlatRepresentationTransformerModel(torch.nn.Module):
    def __init__(self, n_layers, n_heads, hidden_size):
        super(FlatRepresentationTransformerModel, self).__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hidden_size = hidden_size

        self.feature_encoder = torch.nn.Linear(2, hidden_size)
        self.string_embedding_layer = torch.nn.Embedding(6, hidden_size)
        self.time_encoder = torch.nn.Linear(1, hidden_size)

        self.transformer_encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=self.hidden_size,   
                nhead=self.n_heads,
                dim_feedforward=self.hidden_size,
                dropout=0.1,
                activation='relu',
                norm_first= True,
                batch_first=True
            ),
            num_layers=self.n_layers
        )
        self.output_layer = torch.nn.Sequential(torch.nn.Linear(self.hidden_size, 4))
    
    def forward(self, inputs):
        midi_pitch = inputs["midi_pitch_scaled"]
        midi_velocity = inputs["midi_pseudo_velocity"]
        batch, channel, t, _ = midi_pitch.shape
        
        # expand time
        midi_pitch = einops.rearrange(midi_pitch, 'b c t ft -> b (c t) ft')
        midi_velocity = einops.rearrange(midi_velocity, 'b c t ft -> b (c t) ft')

        ft = torch.cat([midi_pitch, midi_velocity], dim=-1)
        ft_z = self.feature_encoder(ft)

        string_index = torch.arange(0, channel, device=ft_z.device)[:,None]
        string_z = self.string_embedding_layer(string_index)
        string_z  = einops.repeat(string_z, 'c 1 ft-> b (c t) ft', b=batch, t=t, ft=ft_z.shape[-1], c=channel)

        time_z = self.time_encoder(torch.arange(0, t, device=ft_z.device)[:,None].to(midi_pitch.dtype))
        time_z = einops.repeat(time_z, 't ft-> b (c t) ft', c=channel, b=batch)

        z = ft_z + string_z + time_z

        out = self.transformer_encoder(z)
        out = self.output_layer(out)

        out = einops.rearrange(out, 'b (c t) ft -> b c t ft', c=channel, t=t)

        output={}

        output["hex_f0_scaled_hat"] = torch.sigmoid(out[...,0].unsqueeze(-1))
        output["hex_loudness_scaled_hat"] = torch.nn.functional.softplus(out[...,1].unsqueeze(-1))
        output["hex_periodicity_hat"] = torch.sigmoid(out[...,2].unsqueeze(-1))
        output["hex_centroid_scaled_hat"] = torch.sigmoid(out[...,3].unsqueeze(-1))
        return output