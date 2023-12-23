#%%
from train_control import ControlBase
import torch
from util import play_audio, forward_fill_midi_pitch, midi_to_unit, linear_quantize, linear_dequantize, features_to_notes, unit_to_midi
from matplotlib import pyplot as plt
from data import GuitarSetDataset
import einops
from util import gaussian_window, viterbi, convert_dtype
import numpy as np
from train_synthesis import SynthesisBase
import data
import preprocessing
import tqdm

#ckpt='artefacts/control_checkpoints/true-disco-546.ckpt'
#ckpt = 'artefacts/control_checkpoints/rural-vortex-548.ckpt'
#ckpt='artefacts/control_checkpoints/floral-voice-571.ckpt'

# 64
ckpt='artefacts/listening_test_checkpoints/dry-wind-611.ckpt'
synthesis_ckpt = "artefacts/synthesis_checkpoints/tough-cosmos-160.ckpt"

# 128
ckpt = 'artefacts/listening_test_checkpoints/balmy-durian-637.ckpt'
synthesis_ckpt = "artefacts/synthesis_checkpoints/proud-donkey-161.ckpt"

# 128
ckpt = 'artefacts/listening_test_checkpoints/worthy-durian-700.ckpt'
ckpt = "artefacts/listening_test_checkpoints/lyric-disco-751-epoch-21.ckpt"
#ckpt = 'artefacts/control_checkpoints/peach-totem-722.ckpt'
#ckpt = 'artefacts/control_checkpoints/misunderstood-sun-720.ckpt'
synthesis_ckpt = "default"

model = ControlBase.load_from_checkpoint(ckpt)

#%% get pitch and velocity distributions
# load trn dataset
trn_ds = data.load_prepared_data(prepared_data_path="./artefacts/guitarset_dataset_data_trn.pt", seconds_per_clip="full", sample_rate=model.config["model_sample_rate"], feature_frame_rate=model.config["model_ft_frame_rate"], use_random_offset=False)
trn_dl = torch.utils.data.DataLoader(trn_ds, batch_size=1, shuffle=False, num_workers=0)

#%%
midi_pitch = []
midi_velocity = []

for sample in tqdm.tqdm(trn_dl):
    sample = preprocessing.preprocess_model_inputs(sample)
    sample = convert_dtype(sample,{torch.float32:model.dtype, torch.float64:model.dtype}) 
    midi_pitch.append(sample["midi_pitch_scaled"].flatten())
    midi_velocity.append(sample["midi_pseudo_velocity"].flatten())

#%%
midi_pitch = torch.cat(midi_pitch, dim=0)
midi_velocity = torch.cat(midi_velocity, dim=0)
# for sample in tqdm.tqdm(trn_dl):
#     notes = features_to_notes(sample)
#     pitches += [note["midi_pitch"] for note in notes]
#     velocities += [note["midi_pseudo_velocity"] for note in notes]

#%%
qpitch = model.input_quantizers["midi_pitch_scaled"].quantize(midi_pitch)

unique_pitches, counts = torch.unique(qpitch, return_counts=True)

#%%
for pitch, count in zip(unique_pitches, counts):
    print(f"{pitch.item():.2f} {count.item():.2f}")

#%%
def correct_pitch(midi_pitch_scaled, unique_pitches, counts, min_count):
    frequent_pitches = unique_pitches[counts>min_count]
    # dequantize
    frequent_pitches = model.input_quantizers["midi_pitch_scaled"].dequantize(frequent_pitches)

    # replace pitch with closest frequent pitch
    midi_pitch_scaled = midi_pitch_scaled.clone()
    for string in range(6):
        for frame in range(midi_pitch_scaled.shape[-1]):
            pitch = midi_pitch_scaled[0,string,frame]
            # find closest frequent pitch
            closest_pitch = frequent_pitches[torch.argmin(torch.abs(frequent_pitches - pitch))]
            midi_pitch_scaled[0,string,frame] = closest_pitch
    

    return midi_pitch_scaled    
    
# correct_pitch(unique_pitches, counts)
#%%
# remove 0

# plt hist of pitch using quantized pitch, using each pitch as a bin
plt.hist(qpitch, bins=unique_pitches)
plt.show()




#%%
#%%

model.input_quantizers

print(model.input_quantizers["midi_pitch_scaled"].values)
# histogram values
for input_quantizer in model.input_quantizers.values():
    plt.hist(input_quantizer.values)
    plt.show()

# print number of epochs
print(model.current_epoch)
# print number of epochs
print(model.learning_rate)
# print feature frame rate
print(model.config["model_ft_frame_rate"])
# print start time
print(model.config)

if synthesis_ckpt != "default":
    model.synthesis_model = SynthesisBase.load_from_checkpoint(synthesis_ckpt)
print(model.config)
N_FRAMES = model.config["n_seconds"] * model.config["model_ft_frame_rate"]

# seed
torch.manual_seed(0)
np.random.seed(0)

tst_ds = data.load_prepared_data(prepared_data_path="./artefacts/guitarset_dataset_data_tst.pt", seconds_per_clip=model.config["n_seconds"], sample_rate = model.config["model_sample_rate"], feature_frame_rate=model.config["model_ft_frame_rate"], use_random_offset=False)

sample_index = 0
dl = torch.utils.data.DataLoader(torch.utils.data.Subset(tst_ds, [sample_index]), batch_size=1, shuffle=True)



#%% export as midi

# get 10th sample
sample = next(iter(dl))

print(sample["midi_pseudo_velocity"].shape)

if model.config["end2end"]:

    sample = preprocessing.preprocess_model_inputs(sample)
    sample = convert_dtype(sample,{torch.float32:model.dtype, torch.float64:model.dtype}) 


    output =  model(sample)
    synthesis_output = model.render(output)["output"]
    play_audio(synthesis_output.detach().numpy().flatten(),model.config["model_sample_rate"]) 

else:
    control  = model.midi2control(sample)

    sample = preprocessing.preprocess_model_inputs(sample)
    sample = convert_dtype(sample,{torch.float32:model.dtype, torch.float64:model.dtype}) 


    # preprocess
  
    #sample["midi_pseudo_velocity"]=torch.clamp(sample["midi_pseudo_velocity"], 1, 1.1)

    print(sample.keys())
    # for feature in model.config["classification_features"]:
    #     feature_name = feature["name"]
    #     #if feature_name != "hex_f0_scaled":
    #     probs = torch.softmax(control[feature_name+"_logits"], dim=-1)
    #     #probs = viterbi(probs, proximity_prior_sigma=0.05)
    #     control[feature_name] = linear_dequantize( torch.argmax(probs, dim=-1), feature["range"][0], feature["range"][1], feature["n_bins"])[...,None]

    for key in sample:
        # if tensor, print shape and plot
        if isinstance(sample[key], torch.Tensor):
            print(key, sample[key].shape)
            if len(sample[key].shape)>1 and sample[key].shape[1]==6:
                print(key, sample[key].shape)
                for string in range(6):
                    plt.plot(sample[key][0,string,:].detach().numpy())
                plt.title(key)
                plt.show()

    output =  model.control2audio(sample)
    play_audio(output["output"].detach().numpy().flatten(), model.synthesis_model.sample_rate)

    output =  model.control2audio(control)
    play_audio(output["output"].detach().numpy().flatten(), model.synthesis_model.sample_rate)


pitch = torch.zeros(1, 6, N_FRAMES)
velocity = torch.ones(1, 6, N_FRAMES)

# lowest note is -4
STRING_INDEX = 2
steps = 32
note_length_ratio = 0.9
PITCH = 47+5*STRING_INDEX
VELOCITY = 1.05
melody = [ p + PITCH for p in [4,3,0,7,3,5,5,7,0,1,2,3,6,5,5,5,5]]    

#melody = [ p + PITCH for p in [0, 0, 0,0,0,0,0]]
step_length = N_FRAMES // steps

midi_activities = torch.zeros(steps, step_length) * 0
midi_activities[:, :int(step_length * note_length_ratio)] = 1
midi_activities = midi_activities.flatten()
midi_activity = torch.zeros(1, 6, N_FRAMES)
midi_activity[:,STRING_INDEX,:] = midi_activities

melody = (melody * steps)[:steps]
# repeat melody to match steps
melody = torch.tensor(melody)[:,None].repeat(1, step_length)
pitches =  melody
#pitches = torch.ones(steps, step_length) * PITCH
pitches[:, int(step_length * note_length_ratio):] = 0
pitches = pitches.flatten()

velocities = torch.ones(steps, step_length) * VELOCITY
velocities[:, int(step_length * note_length_ratio):] = 1
velocities = velocities.flatten()

voice_index = torch.arange(6)[None, :, None]

pitch[:,STRING_INDEX,:] = pitches

#pitch = forward_fill_midi_pitch(pitch[0])[None,...]

velocity[:,STRING_INDEX,:] = velocities

voice_index = torch.arange(6)[None, :, None]

inputs = {"midi_pitch": pitch, "midi_pseudo_velocity": velocity, "string_index": voice_index}


sample = inputs
for key in sample:
    # if tensor, print shape and plot
    if isinstance(sample[key], torch.Tensor) and "midi" in key:
        print(key, sample[key].shape)
        for string in range(6):
            plt.plot(sample[key][0,string,:].detach().numpy())
        plt.title(key)
        plt.show()

if model.config["end2end"]:
    sample = preprocessing.preprocess_model_inputs(sample)
    inputs = convert_dtype(sample,{torch.float32:model.dtype, torch.float64:model.dtype}) 

    #inputs["midi_pitch_scaled"] = correct_pitch(inputs["midi_pitch_scaled"], unique_pitches, counts, 10000)
    output =  model(inputs)
    synthesis_output = model.render(output)["output"]
    play_audio(synthesis_output.detach().numpy().flatten(),model.config["model_sample_rate"])

else:
    control  = model.midi2control(inputs)

    hex_pitch_logits = control["hex_f0_scaled_logits"]
    # # only select STRING INDEX but keep dimension
    hex_pitch_probs = torch.softmax(hex_pitch_logits, dim=-1)

    # # plot pitch probs heatmap and midi pitch contour on top, scale y axis to match image height (0,1)
    plt.imshow(einops.rearrange(hex_pitch_probs, "b v t f -> t (b v f)").detach().numpy().T, aspect="auto", origin="lower")
    plt.show()


    for feature in model.config["classification_features"]:
        feature_name = feature["name"]
        #if feature_name != "hex_f0_scaled":
        probs = torch.softmax(control[feature_name+"_logits"], dim=-1)
        #probs = viterbi(probs, proximity_prior_sigma=0.05)
        control[feature_name] = linear_dequantize( torch.argmax(probs, dim=-1), feature["range"][0], feature["range"][1], feature["n_bins"])[...,None]

    LOCK_TO_MIDI_PITCH = False
    if LOCK_TO_MIDI_PITCH:
        pitch_feature_metadata = [feature for feature in model.config["classification_features"] if "hex_f0_scaled" in feature["name"]][0]
        midi_pitch_map = linear_quantize(control["midi_pitch_scaled"], pitch_feature_metadata["range"][0], pitch_feature_metadata["range"][1], pitch_feature_metadata["n_bins"])
        midi_pitch_map = torch.nn.functional.one_hot(midi_pitch_map, pitch_feature_metadata["n_bins"]).float()
        # alpha = 1.0
        adjusted_pitch_probs =  midi_pitch_map * midi_activity[...,None] + hex_pitch_probs * (1 - midi_activity[...,None])
        control["hex_f0_scaled"] = linear_dequantize(torch.argmax(adjusted_pitch_probs, dim=-1), pitch_feature_metadata["range"][0], pitch_feature_metadata["range"][1], pitch_feature_metadata["n_bins"])[...,None]

    # plt.imshow( einops.rearrange(control["hex_loudness_scaled_logits"], "b v t f -> t (b v f)").detach().numpy().T, aspect="auto", origin="lower")
    # plt.show()

    # print(midi_pitch_map.shape)
    # plt.imshow(einops.rearrange(midi_pitch_map, "b v t f -> t (b v f)").detach().numpy().T, aspect="auto", origin="lower")
    # plt.show()


    # # take mean of midi pitch map and midi activity


    # plt.imshow(einops.rearrange(adjusted_pitch_probs, "b v t f -> t (b v f)").detach().numpy().T, aspect="auto", origin="lower")
    # plt.show()

    # print(control.keys())


    # # plot f0, centroid, loudness, periodicity
    # plt.figure(figsize=(20,10))
    # plt.subplot(4,1,1)
    # plt.title("f0")
    # plt.plot(control["hex_f0_scaled"][0,STRING_INDEX].detach().numpy())
    # plt.subplot(4,1,2)
    # plt.plot(control["hex_centroid_scaled"][0,STRING_INDEX].detach().numpy())
    # plt.title("centroid")
    # plt.subplot(4,1,3)
    # plt.plot(control["hex_loudness_scaled"][0,STRING_INDEX].detach().numpy())
    # plt.title("loudness")
    # plt.subplot(4,1,4)
    # plt.title("periodicity")
    # plt.plot(control["hex_periodicity"][0,STRING_INDEX].detach().numpy())


    # loudness_alpha = 0.8
    # for t in range(1, control["hex_loudness_scaled"].shape[-2]):
    # loudness_alpha = 0.8
    # for t in range(1, control["hex_loudness_scaled"].shape[-2]):
    #     control["hex_loudness_scaled"][:,:,t,:] = control["hex_loudness_scaled"][:,:,t,:] * loudness_alpha + control["hex_loudness_scaled"][:,:,t-1,:] * (1 - loudness_alpha)
    # # reverse order
    # for t in reversed(range(0, control["hex_loudness_scaled"].shape[-2]-1)):
    #     control["hex_loudness_scaled"][:,:,t,:] = control["hex_loudness_scaled"][:,:,t,:] * loudness_alpha + control["hex_loudness_scaled"][:,:,t+1,:] * (1 - loudness_alpha)

    # centroid_alpha = 0.8
    # for t in range(1, control["hex_centroid_scaled"].shape[-2]):
    #     control["hex_centroid_scaled"][:,:,t,:] = control["hex_centroid_scaled"][:,:,t,:] * centroid_alpha + control["hex_centroid_scaled"][:,:,t-1,:] * (1 - centroid_alpha)
    # for t in reversed(range(0, control["hex_centroid_scaled"].shape[-2]-1)):
    #     control["hex_centroid_scaled"][:,:,t,:] = control["hex_centroid_scaled"][:,:,t,:] * centroid_alpha + control["hex_centroid_scaled"][:,:,t+1,:] * (1 - centroid_alpha)

    # periodicity_alpha = 0.8
    # for t in range(1, control["hex_periodicity"].shape[-2]):
    #     control["hex_periodicity"][:,:,t,:] = control["hex_periodicity"][:,:,t,:] * periodicity_alpha + control["hex_periodicity"][:,:,t-1,:] * (1 - periodicity_alpha)
    # for t in reversed(range(0, control["hex_periodicity"].shape[-2]-1)):
    #     control["hex_periodicity"][:,:,t,:] = control["hex_periodicity"][:,:,t,:] * periodicity_alpha + control["hex_periodicity"][:,:,t+1,:] * (1 - periodicity_alpha)

    # plot f0, centroid, loudness, periodicity
    plt.figure(figsize=(20,10))
    plt.subplot(4,1,1)
    plt.title("f0")
    plt.plot(control["hex_f0_scaled"][0,STRING_INDEX].detach().numpy())
    plt.subplot(4,1,2)
    plt.plot(control["hex_centroid_scaled"][0,STRING_INDEX].detach().numpy())
    plt.title("centroid")
    plt.subplot(4,1,3)
    plt.plot(control["hex_loudness_scaled"][0,STRING_INDEX].detach().numpy())
    plt.title("loudness")
    plt.subplot(4,1,4)
    plt.title("periodicity")
    plt.plot(control["hex_periodicity"][0,STRING_INDEX].detach().numpy())

    plt.show()

    output =  model.control2audio(control)

    # play audio
    play_audio(output["output"].detach().numpy().flatten(), model.synthesis_model.sample_rate)

    # downsample to 16kHz and play
    play_audio(output["output"].detach().numpy().flatten()[::model.synthesis_model.sample_rate//16000], 16000)
    # print


# %%
