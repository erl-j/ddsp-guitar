import penn
import torchcrepe
from util import resample_feature
import torch
import librosa
import torchaudio
import numpy as np

def compute_spectral_centroid(hex_y, win_length, hop_length, sample_rate):
    return torchaudio.functional.spectral_centroid(hex_y,n_fft=win_length, win_length=win_length, hop_length=hop_length, sample_rate=sample_rate, window = torch.hann_window(win_length), pad=0)

# from @caillonantoine ircarm/ddsp_pytorch, used to verify the pure pytorch implementation below
def compute_loudness_ptddsp(signal, sampling_rate, block_size, n_fft):
    S = librosa.stft(
        signal,
        n_fft=n_fft,
        hop_length=block_size,
        win_length=n_fft,
        center=True,
        pad_mode="constant",
        window= "hann"
    )
    print(S.shape)
    print(S.mean(0))
    S = np.log(abs(S) + 1e-7)
    f = librosa.fft_frequencies(sampling_rate, n_fft)
    a_weight = librosa.A_weighting(f)
    S = S + a_weight.reshape(-1, 1)
    S = np.mean(S, 0)
    print(S.shape)
    return S

# pure pytorch implementation of compute_loudness. 
# adapted from @caillonantoine ircarm/ddsp_pytorch
def compute_loudness(signal,sample_rate,hop_length,n_fft):
    S = torch.stft(
        signal,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        center=True,
        return_complex=True,
        pad_mode="constant",
        window=torch.hann_window(n_fft)
    )
    S = torch.log(torch.abs(S) + 1e-7)
    f = librosa.fft_frequencies(sample_rate, n_fft)
    a_weight = librosa.A_weighting(f)

    S = S + a_weight.reshape(1,-1,1)
    S =torch.mean(S, 1)
    return S

def compute_rms(y, window_size,hop_frames):
    # warn if hop_frames is larger than window_size
    if hop_frames > window_size:
        print("Warning: hop_frames is larger than window_size. This may cause unexpected behavior.")
    rms = []
    for i in range(y.shape[0]):
        frames = y[i].unfold(0, window_size, hop_frames)
        ld = torch.sqrt(torch.mean(frames**2, dim=1))
        ld = ld[None,...]
        rms.append(ld)
    rms = torch.cat(rms, dim=0)
    return rms

def compute_pitch(y, sample_rate,fmin, fmax, hop_frames, device=None,batch_size=None,pad=False, return_probabilities=False):
    PITCH_EXTRACTION_ALGORITHM = "crepe"
    pitch = []
    periodicity = []
    probabilities = []

    if PITCH_EXTRACTION_ALGORITHM=="penn":
        gpu=device
        checkpoint=penn.DEFAULT_CHECKPOINT
        interp_unvoiced_at = None
        hopsize_s=hop_frames/sample_rate

        for i in range(y.shape[0]):
            p,p2 = penn.from_audio(
                y[i][None,...],
                sample_rate,
                hopsize=hopsize_s,
                fmin=fmin,
                fmax=fmax,
                checkpoint=checkpoint,
                batch_size=batch_size,
                pad=pad,
                interp_unvoiced_at=interp_unvoiced_at,
                gpu=gpu)

            pitch.append(p.cpu())
            periodicity.append(p2.cpu())

    elif PITCH_EXTRACTION_ALGORITHM=="crepe":
        # Here we'll use a 5 millisecond hop length
        hop_frames= hop_frames
        # Provide a sensible frequency range for your domain (upper limit is 2006 Hz)
        # This would be a reasonable range for speech
    
        # Select a model capacity--one of "tiny" or "full"
        model = 'full'
        # Choose a device to use for inference
        # Pick a batch size that doesn't cause memory errors on your gpu
        # Compute pitch using first gpu
        for i in range(y.shape[0]):
            if return_probabilities:
                p,p2, probs = torchcrepe.predict(
                                        y[i][None,...],
                                        sample_rate,
                                        hop_frames,
                                        fmin,
                                        fmax,
                                        model,
                                        batch_size=batch_size,
                                        device=device,
                                        return_periodicity=True,
                                        return_probabilities=return_probabilities,
                                        pad=pad
                )
                probabilities.append(probs)
            else:
                p,p2= torchcrepe.predict(
                                        y[i][None,...],
                                        sample_rate,
                                        hop_frames,
                                        fmin,
                                        fmax,
                                        model,
                                        batch_size=batch_size,
                                        device=device,
                                        return_periodicity=True,
                                        pad=pad
                )
            pitch.append(p.cpu())
            periodicity.append(p2.cpu())
        


    pitch = torch.cat(pitch, dim=0)
    periodicity = torch.cat(periodicity, dim=0)

    if return_probabilities:
        pitch, periodicity, probabilities

    return pitch, periodicity

def median_filtering(x, window_size):
    new_x = x.clone()
    x = x.clone()
    # pad x
    x = torch.nn.functional.pad(x, (window_size//2,window_size//2), mode='reflect')
    for i in range(new_x.shape[1]):
        new_x[:,i] = torch.median(x[:,i:i+window_size],dim=1)[0]
    return new_x
    
def compute_pseudo_velocity(midi_activity, audio_loudness):
    assert midi_activity.shape == audio_loudness.shape
    # for every midi pitch, get the max loudness
    midi_loudness = torch.zeros_like(midi_activity)
    note_start_stops = []
    is_active = False
    for i in range(midi_activity.shape[0]):
        if midi_activity[i]:
            if not is_active:
                note_start_stops.append([i])
                is_active = True
        else:
            if is_active:
                note_start_stops[-1].append(i)
                is_active = False

    # Check if the last note is still active at the end
    if is_active:
        note_start_stops[-1].append(midi_activity.shape[0])

    for note_start_stop in note_start_stops:
        midi_loudness[note_start_stop[0]:note_start_stop[1]] = torch.max(audio_loudness[note_start_stop[0]:note_start_stop[1]])
    return midi_loudness