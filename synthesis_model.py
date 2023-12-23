import einops
import numpy as np
import torch

import nn
import synth
from nn import RNN, DilatedConvStackStack, FcStack
from util import fold, hz_to_unit, unfold, unit_to_hz


class DDSPDecoderAdapter(torch.nn.Module):
    def __init__(self,
            model,
            input_splits=(),
            output_splits=(),
            is_wavenet=False,
            ):

        super().__init__()
        self.input_splits = input_splits
        self.output_splits = output_splits

        self.is_wavenet = is_wavenet

        self.n_input_channels = sum([v[1] for v in input_splits if v[0] != 'voice_embedding'])
        self.n_output_channels = sum([v[1] for v in output_splits])

        self.model = model
      
    def forward(self, inputs):
        batch_size,n_channels,n_frames,_= inputs['hex_f0_scaled'].shape
        voice_embedding = inputs['voice_embedding']
        inputs_wo_voice_embedding = {k: v for k, v in inputs.items() if k != 'voice_embedding'}
        inputs_merged = torch.concat(tuple(inputs_wo_voice_embedding.values()), axis=-1)
        if self.is_wavenet:
            x = fold(inputs_merged)
            cond = fold(voice_embedding)
            x = einops.rearrange(x, 'bc t ft -> bc ft t')
            cond = einops.rearrange(cond, 'bc t ft -> bc ft t')
            x = self.model(x, cond)
            x = einops.rearrange(x, 'bc ft t -> bc t ft')
            x = unfold(x,n_channels = n_channels)
        else:
            x = torch.cat((inputs_merged, voice_embedding), axis=-1)
            x = self.model(x)
        output_dict = { name: x[..., start:start+dim] for (name, dim), start in zip(self.output_splits, np.cumsum([0] + [v[1] for v in self.output_splits])[:-1]) }
        return output_dict

class DDSPModel(torch.nn.Module):
    def __init__(self,
                sample_rate,
                n_harmonics,
                n_noise_bands,
                ir_duration,
                use_one_ir_per_voice,
                input_ft_splits,
                get_decoder,
                min_f0_hz,
                max_f0_hz,
                voice_embedding_size,
                noise_bias,
                n_voices) -> None:
        super().__init__() 


        if voice_embedding_size == -1:
            self.voice_embedding_layer = lambda index_tensor :  torch.nn.functional.one_hot(index_tensor, n_voices).float()
            voice_embedding_size = n_voices
        else:
            self.voice_embedding_layer = torch.nn.Embedding(n_voices, voice_embedding_size)

        self.input_ft_splits_w_voice_embedding = input_ft_splits + (("voice_embedding", voice_embedding_size),)
        self.sample_rate = sample_rate
        self.min_f0 = min_f0_hz
        self.max_f0 = max_f0_hz
        self.harmonic_synth = synth.HarmonicSynth(sample_rate=self.sample_rate)
        self.noise_synth = synth.FilteredNoiseSynth(sample_rate=self.sample_rate)
        self.use_one_ir_per_voice = use_one_ir_per_voice
        if self.use_one_ir_per_voice:
            self.reverb = synth.MultiChannelReverb(sample_rate=self.sample_rate, ir_duration=ir_duration, n_channels=n_voices)
        else:
            self.reverb = synth.Reverb(sample_rate=self.sample_rate, ir_duration=ir_duration)

        self.decoder_output_splits=(
            ("harmonic_partial_amp_output", n_harmonics),
            ("harmonic_global_amp_output", 1),
            ("noise_band_amp_output", n_noise_bands),
        )
        self.decoder = get_decoder(input_ft_splits, self.decoder_output_splits)
        self.noise_bias = noise_bias

    def synthesize(self, outputs, n_samples):
        batch_size,channels,n_feature_frames,_= outputs["hex_f0_scaled"].shape

        # scale and fold synth parameters
        f0_hz = unit_to_hz(outputs["hex_f0_scaled"], hz_min=self.min_f0, hz_max=self.max_f0)
        f0_hz = fold(f0_hz)

        noise_amp = torch.nn.functional.softplus(outputs["noise_band_amp_output"]+self.noise_bias)
        noise_amp = fold(noise_amp)

        harm_amps = torch.sigmoid(outputs["harmonic_partial_amp_output"])
        harm_amps = harm_amps / (harm_amps.sum(dim=-1, keepdim=True)+1e-8)
        harm_amps = fold(harm_amps)
        
        global_amp = torch.nn.functional.softplus(outputs["harmonic_global_amp_output"])
        global_amp = fold(global_amp)

        # save synth parameters for logging
        outputs["harm_amps"] = harm_amps
        outputs["global_amp"] = global_amp
        outputs["f0_hz"] = f0_hz
        outputs["noise_amps"] = noise_amp

        # synthesize audio
        noise_synth_output = self.noise_synth(noise_amp, n_samples)
        noise_synth_output = unfold(noise_synth_output, n_channels=channels)

        harmonic_synth_output = self.harmonic_synth(f0_hz, harm_amps, global_amp, n_samples)
        harmonic_synth_output = unfold(harmonic_synth_output, n_channels=channels)
        
        # crop noise synth output to match harmonic synth output evenly on each side
        excess_samples = noise_synth_output.shape[2] - harmonic_synth_output.shape[2]
        if excess_samples > 0:
            noise_synth_output = noise_synth_output[:,:,excess_samples//2:-excess_samples//2]

        dry_mix_output = harmonic_synth_output + noise_synth_output
        
        # save intermediate audio outputs for logging
        outputs["string_harmonic_output"] = harmonic_synth_output
        outputs["string_noise_output"] = noise_synth_output
        outputs["string_dry_output"] = dry_mix_output
        outputs["dry_mix"] = torch.sum(dry_mix_output,dim=1, keepdim=True)

        # apply reverb
        if self.use_one_ir_per_voice:
            outputs["reverb_mix"] = self.reverb(dry_mix_output)
            outputs["isolated_reverb_mix"] = self.reverb.forward_without_channel_summation(dry_mix_output)
        else:
            outputs["reverb_mix"]=self.reverb(outputs["dry_mix"][:,0,:]).unsqueeze(1)

        outputs["ir"]=self.reverb.get_ir()
        outputs["output"] = outputs["dry_mix"]+outputs["reverb_mix"]

        outputs["string_output"] = outputs["string_dry_output"]+outputs["reverb_mix"]
        outputs["isolated_string_output"] = outputs["string_dry_output"]+outputs["isolated_reverb_mix"]
        

        return outputs

    def forward(self,inputs, n_samples):
        batch_size,channels,n_feature_frames,_= inputs["hex_f0_scaled"].shape
        # add voice embedding to inputs
        voice_embedding = self.voice_embedding_layer(inputs["voice_index"]).squeeze(2)
        inputs["voice_embedding"] = voice_embedding.expand(batch_size, -1, n_feature_frames, -1)

        # prepare decoder inputs
        decoder_inputs={}
        for ft_name, ft_n_channels in self.input_ft_splits_w_voice_embedding:
            # if ft_name not in inputs:
            #     raise ValueError(f"Missing input feature {ft_name}")
            # assert inputs[ft_name].shape[-1] == ft_n_channels, f"Input feature {ft_name} has wrong number of channels"
            decoder_inputs[ft_name] = inputs[ft_name]

        outputs = self.decoder(decoder_inputs)

        return self.synthesize({**inputs,**outputs}, n_samples)


class MixFcDecoder(torch.nn.Module):
  """
  Unlike the ddsp decoder, this one does not have an input stack per input feature but instead mixes all inputs into one stack.
  We also sum the rnn output with the input stack output instead of concatenating them.
  """

  def __init__(self,
               rnn_channels=512,    
               rnn_type='gru',
               ch=512,
               layers_per_stack=3,
               rnn_layers=1,
               input_splits=(),
               output_splits=()):
    """Constructor.
    Args:
        rnn_channels: Dims for the RNN layer.
        rnn_type: Either 'gru' or 'lstm'.
        ch: Dims of the fully connected layers.
        layers_per_stack: Fully connected layers per a stack.
        input_splits: List of (name, dims) tuples for the inputs.
        output_splits: List of (name, dims) tuples for the outputs.
    """
    super().__init__()
    self.input_splits = input_splits
    self.output_splits = output_splits
    self.rnn_channels = rnn_channels
    self.input_size = sum([v[1] for v in input_splits])
    # Layers.
    self.input_stack = FcStack(self.input_size,rnn_channels,layers_per_stack)
    self.rnn = RNN(rnn_channels, rnn_type, bidirectional=True, return_sequences=True, n_layers=rnn_layers)
    self.out_stack = FcStack(rnn_channels, ch, layers_per_stack)
    n_out = sum([v[1] for v in output_splits])
    self.dense_out = torch.nn.Linear(rnn_channels, n_out)

  def forward(self, inputs):
    batch_size,n_channels,n_frames,_= inputs['hex_f0_scaled'].shape

    inputs_merged = torch.concat([inputs[ft] for ft, _ in self.input_splits], axis=-1)

    inputs_merged_p= self.input_stack(inputs_merged)

    inputs_merged = inputs_merged_p+inputs["voice_embedding"]
 
    x = einops.rearrange(inputs_merged_p, 'b c t ft -> (b c) t ft')
    # Run an RNN over the latents.
    x = self.rnn(x)
    x = einops.rearrange(x, '(b c) t ft -> b c t ft', b=batch_size, c=n_channels)
    # sum outputs from each direction   
    x = x[..., :self.rnn_channels] + x[..., self.rnn_channels:]
    #x = torch.cat([inputs_merged, x], dim=-1)
    x = x + inputs_merged
    # Final processing.
    x = self.out_stack(x)
    x = self.dense_out(x)
    # assert no nans or infs
    assert not torch.isnan(x).any()
    assert not torch.isinf(x).any()

    output_dict = { name: x[..., start:start+dim] for (name, dim), start in zip(self.output_splits, np.cumsum([0] + [v[1] for v in self.output_splits])[:-1]) }
    return output_dict

#   self.decoder = MixFcDecoder(
#             rnn_channels=rnn_channels,
#             layers_per_stack=layers_per_stack,
#             rnn_type=rnn_type,
#             ch = fc_channels,
#             input_splits = self.input_ft_splits,
#             output_splits=(
#                 ("harmonic_partial_amp_output", n_harmonics),
#                 ("harmonic_global_amp_output", 1),
#                 ("noise_band_amp_output", n_noise_bands),
#             ))

# self.wavenet = WaveNet(  
# input_channels=n_input_channels,
# output_channels=n_output_channels,
# residual_channels=wavenet_args['residual_channels'],
# activation=wavenet_args['activation'],
# dilations = wavenet_args['dilations'],
# cond_channels = wavenet_args['cond_channels'],
# skip_channels=wavenet_args['skip_channels'],
# kernel_size=wavenet_args['kernel_size'],
# causal = wavenet_args['causal'],
# )