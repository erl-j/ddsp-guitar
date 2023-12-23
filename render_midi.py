# %%
import pretty_midi
import torch
import tqdm
import soundfile as sf
import preprocessing
from train_control import ControlBase
from util import (
    convert_dtype,
    play_audio,
)
import math
import numpy as np
from demo_utils import pretty_midi_to_pitch_vel, linear_interpolate_list, remove_out_of_range_notes, crop_pretty_midi
import argparse
import os
#%%
parser = argparse.ArgumentParser()

parser.add_argument("--midi_path", type=str, default=None, required=True, help="path to midi file")
parser.add_argument("--output_path", type=str, default=None, help="path to output wav file")
parser.add_argument("--crop-seconds", type=int, default=None, help="crop midi to this many seconds")
parser.add_argument("--device", type=str, default="cuda:0", help="device to run on")
parser.add_argument("--pitch-correction", action="store_true", default=True, help="replace predicted pitch with midi pitch on active notes")
parser.add_argument("--legato", action="store_true", default=False, help="use legato, otherwise forces staccato on subsequent notes on same string")

args = parser.parse_args()

midi_path = "test_midi/never_meant.mid"

ckpt = "checkpoints/unified.ckpt"
os.makedirs("checkpoints", exist_ok=True)
# if checkpoint doesn't exist, download from huggingface
if not os.path.exists(ckpt):
    # download from huggingface
    os.system(f"wget https://huggingface.co/erl-j/ddsp-guitar-unified/resolve/main/unified.ckpt?download=true -O {ckpt}")
    
#%%
device = torch.device(args.device if torch.cuda.is_available() else "cpu")

model = ControlBase.load_from_checkpoint(ckpt)
model = model.to(device)
model.eval()
model_ft_frame_rate = model.config["model_ft_frame_rate"]

#%%
pm = pretty_midi.PrettyMIDI(midi_path)
if args.crop_seconds is not None:
    pm = crop_pretty_midi(pm, args.crop_seconds)
pm = remove_out_of_range_notes(pm)

def extend_notes(pm):
    for instrument in pm.instruments:
        for note in instrument.notes:
            note.end = 4 * (note.end - note.start) + note.start
    return pm

pm = extend_notes(pm)

#%%
seconds = pm.get_end_time()

pitch, velocity = pretty_midi_to_pitch_vel(pm, model_ft_frame_rate, legato=args.legato)
# %%
MODEL_FRAMES = model.config["n_seconds"] * model.config["model_ft_frame_rate"]

# pad to multiple of model frames
frames_before_padding = pitch.shape[-1]
total_frames_w_padding = math.ceil(pitch.shape[-1] / MODEL_FRAMES ) * MODEL_FRAMES
pad = total_frames_w_padding - pitch.shape[-1]
pitch = torch.nn.functional.pad(pitch, (0, pad))
velocity = torch.nn.functional.pad(velocity, (0, pad))

SKIP_RATIO = 0.5
skip_frames = int(MODEL_FRAMES * SKIP_RATIO)
voice_index = torch.arange(6)[None, :, None]

print("generating synth params")
# predict synth params for each window
synth_params = []
for window_start in tqdm.tqdm(range(0, pitch.shape[-1], skip_frames)):
    window_end = window_start + MODEL_FRAMES
    pitch_window = pitch[..., window_start:window_end]
    velocity_window = velocity[..., window_start:window_end]

    inputs = {
        "midi_pitch": pitch_window[None, ...],
        "midi_pseudo_velocity": velocity_window[None, ...],
        "string_index": voice_index
    }

    inputs = preprocessing.preprocess_model_inputs(inputs)
    inputs = convert_dtype(
        inputs, {torch.float32: model.dtype, torch.float64: model.dtype}
    )
    # put on gpu
    inputs = {k: v.to(device) for k, v in inputs.items()}
    synth_param = model(inputs)

    # replace prediction with ground truth on active notes
    if args.pitch_correction:
        synth_param["hex_f0_scaled_hat"][inputs["midi_pseudo_velocity"] != 1] = inputs["midi_pitch_scaled"][inputs["midi_pseudo_velocity"] != 1]
    synth_params.append(synth_param)

#
full_synth_params = {}
for key in ["harmonic_partial_amp_output", "harmonic_global_amp_output", "noise_band_amp_output", "hex_f0_scaled_hat"]:
    full_synth_params[key] = linear_interpolate_list([x[key] for x in synth_params], skip_frames)
    full_synth_params[key] = full_synth_params[key][:, : , :frames_before_padding, :]

# %%
full_synth_params["voice_index"] = voice_index  

for key in full_synth_params:
    full_synth_params[key] = full_synth_params[key].to("cpu")

n_samples = int(seconds * model.config["model_sample_rate"])

print("generating audio from synth params")
synthesis_output = model.to("cpu").render(full_synth_params, n_samples )["output"]
#%%

if args.output_path is None:
    outpath = os.path.splitext(args.midi_path)[0] + ".wav"
else:
    outpath = args.output_path

sf.write(outpath, synthesis_output.detach().cpu().numpy().flatten(), model.config["model_sample_rate"])
# %%
