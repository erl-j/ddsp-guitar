import math
import torch

def pretty_midi_to_pitch_vel(pm, model_ft_frame_rate,legato=False):
    # get duration of midi
    string_to_pitch_mapping = {
        0: range(40, 45),
        1: range(45, 50),
        2: range(50, 55),
        3: range(55, 60),
        4: range(60, 65),
        5: range(65, 89),
    }
    pitch_to_string_mapping = {}
    for string, pitches in string_to_pitch_mapping.items():
        for pitch in pitches:
            pitch_to_string_mapping[pitch] = string
    seconds = pm.get_end_time()
    frames_before_padding = math.ceil(seconds * model_ft_frame_rate)
    pitch = torch.zeros((6, frames_before_padding))
    velocity = torch.zeros((6, frames_before_padding))
    for instrument in pm.instruments:
        for note in instrument.notes:
            start = int(note.start * model_ft_frame_rate)
            end = int(note.end * model_ft_frame_rate)
            string = pitch_to_string_mapping[note.pitch]
            if not legato:
                if start > 1 and pitch[string, start - 1] != 0:
                    # remove legato by setting two previous frames to 0
                    pitch[string, start - 1] = 0
                    velocity[string, start - 1] = 0
                    pitch[string, start - 2] = 0
                    velocity[string, start - 2] = 0
            pitch[string, start:end] = note.pitch
            velocity[string, start:end] = note.velocity
    velocity = midi_to_model_scale_velocity(velocity)
    return pitch, velocity

def linear_interpolate(A, B, skip_frames):
    batch, voice, time, ft = A.shape
    # Extract overlapping parts
    A_overlap = A[:, :, -skip_frames:, :]
    B_overlap = B[:, :, :skip_frames, :]
    # Compute weights for interpolation
    weights = torch.linspace(0, 1, steps=skip_frames).view(1, 1, skip_frames, 1).to(A.device)
    # Interpolate
    interpolated = A_overlap * (1 - weights) + B_overlap * weights
    # Concatenate
    A_non_overlap = A[:, :, :-skip_frames, :]
    B_non_overlap = B[:, :, skip_frames:, :]
    merged = torch.cat([A_non_overlap, interpolated, B_non_overlap], dim=2)
    return merged

def linear_interpolate_list(list, skip_frames):
    a = list[0]
    for i in range(1, len(list)):
        a = linear_interpolate(a, list[i], skip_frames)
    return a

def midi_to_model_scale_velocity(midi_velocity, min_pseudo_vel=1.08, max_pseudo_vel=1.12):
    # 0 maps to 1
    # 1-127 maps to 1.1 - 1.2
    # zero_mask = midi_velocity == 0
    # midi_velocity[~zero_mask] = min_pseudo_vel + (midi_velocity[~zero_mask] / 127) * (max_pseudo_vel - min_pseudo_vel)
    # midi_velocity[zero_mask] = 0
    midi_velocity =  (midi_velocity == 0) + (min_pseudo_vel + (midi_velocity / 127) * (max_pseudo_vel - min_pseudo_vel)) * (midi_velocity != 0)
    return midi_velocity

def remove_out_of_range_notes(pm, min_pitch=40, max_pitch=88):
    # remove drums
    pm.instruments = [i for i in pm.instruments if not i.is_drum]
    for i in range(len(pm.instruments)):
        instrument = pm.instruments[i]
        for j in reversed(range(len(instrument.notes))):
            note = instrument.notes[j]
            if note.pitch < min_pitch or note.pitch > max_pitch:
                instrument.notes.remove(note)
    return pm

def crop_pretty_midi(pm, seconds):
    for i in range(len(pm.instruments)):
        instrument = pm.instruments[i]
        for j in reversed(range(len(instrument.notes))):
            note = instrument.notes[j]
            if note.start > seconds:
                instrument.notes.remove(note)
            elif note.end > seconds:
                note.end = seconds
    return pm