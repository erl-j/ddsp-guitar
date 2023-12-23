#%%
import glob
import os

import jams
import jams.display
import torch
import torchaudio
from tqdm import tqdm

import numpy as np
from feature import compute_pitch, compute_rms, compute_loudness, compute_spectral_centroid, median_filtering
from util import resample_feature, hz_to_unit, forward_fill_midi_pitch
import einops
import data
import util
from feature import compute_pseudo_velocity

GUITAR_F0_MIN_HZ = 35
GUITAR_F0_MAX_HZ = 1200

def replace_nan_with_previous(x):
        x = x.clone()
        if x[0].isnan():
            x[0] = 0
        for i in range(1, x.shape[0]):
            if torch.isnan(x[i]):
                x[i] = x[i-1]
        return x

def parse_guitarset_filename(filename):
        # pattern is <PERFOMER_ID>_<GENRE_ID><CHORD_PROGRESSION_ID>-<BPM>-<KEY>_<SOLO_OR_COMP>.jams
        # e.g. 00_BN1-129-Eb_comp.jams where 00 is performer id, BN is genre, 1 is progression, 129 is bpm, Eb is key, comp means comping
        fields = filename.split("_")
        print(fields)
        performer_id = fields[0]
        genre_progression, bpm,key = fields[1].split("-")
        progression_id = genre_progression[-1]
        genre_id = genre_progression[:-1]
        bpm = int(fields[1].split("-")[1])
        key = fields[1].split("-")[2]
        solo_or_comp = fields[2].split(".")[0]
        return performer_id, genre_id, progression_id, bpm, key, solo_or_comp

def load_prepared_data(prepared_data_path, seconds_per_clip, sample_rate, feature_frame_rate, use_random_offset, pitch_median_filter_window_size=1):
    ds = GuitarSetDataset(None, prepared_path=prepared_data_path)
    ds.set_requested_features(["hex_pitch", "hex_loudness", "hex_periodicity", "hex_centroid_scaled", "string_index", "mic_audio", "hex_audio", "midi_pitch", "midi_pseudo_velocity","midi_activity", "midi_onsets", "midi_offsets", "midi_duration_since_previous_onset"])
    if seconds_per_clip == "full":
        ds.reindex_data_full_clips()
    else:
        ds.reindex_data(seconds_per_clip,samples_per_clip="n_seconds")
    ds.resample_audio(sample_rate)
    ds.repair_spectral_centroid()
    if pitch_median_filter_window_size != 1:
        ds.median_filter_pitch(pitch_median_filter_window_size)
    ds.compute_pseudo_velocity()
    # ds.fill_midi_pitch()
    ds.resample_features(feature_frame_rate)
    ds.use_random_offset(use_random_offset)
    # cache data
    return ds

# contains raw hex data, debleeded data and mic data
# also contains transcription data.
class GuitarSetDataset(torch.utils.data.Dataset):
    def __init__(self, filepaths, prepared_path, sample_rate=None, feature_frame_rate=None, limit_samples=None, pitch_extraction_device="cpu", pitch_extraction_batch_size=None):
        # check that feature frame rate is a divisor of sample rate
        if prepared_path is not None:
            self.load_data(prepared_path)
        else:
            self.sample_rate = sample_rate
            self.pitch_extraction_device = pitch_extraction_device
            self.pitch_extraction_batch_size = pitch_extraction_batch_size
            self.feature_frame_rate = feature_frame_rate
            self.requested_features = "all"
            if filepaths is not None:
                assert sample_rate % feature_frame_rate == 0, "Feature frame rate must be a divisor of sample rate"
                annotation_fps = filepaths
                if limit_samples is not None:
                    annotation_fps = annotation_fps[:limit_samples]
                self.data = []
                for filepath in tqdm(annotation_fps):
                    self.data.append(self.load_example(filepath))
    

    
    def use_random_offset(self, use_random_offset):
        self.use_random_offset = use_random_offset


    def __len__(self):
        return self.len
    
    def load_annotation(self, filepath):
        filename = filepath.split("/")[-1].split(".")[0]
        data_path = "/".join(filepath.split("/")[:-1])
        data_path = data_path.replace("/annotation","")
        annotation_fp = os.path.join(data_path, "annotation", filename + ".jams")
        annotation = jams.load(annotation_fp)
        return annotation

    def load_example(self, filepath):
        sample={}
        print(f"loading {filepath}...")
        
        filename = filepath.split("/")[-1].split(".")[0]
        data_path = "/".join(filepath.split("/")[:-1])
        data_path = data_path.replace("/annotation","")
        print(f"filename: {filename}")
        # performance metadata
        sample["filename"] = filename
        
        performer_id, genre_id, progression_id, bpm, key, solo_or_comp = parse_guitarset_filename(filename+".jams")
        sample["performer_id"] = performer_id
        sample["genre_id"] = genre_id
        sample["progression_id"] = progression_id
        sample["bpm"] = bpm
        sample["key"] = key
        sample["solo_or_comp"] = solo_or_comp

        hop_frames = int(self.sample_rate/self.feature_frame_rate)

        loudness_window_frames = hop_frames * 4

        # hex-pickup_debleeded
        hex_debleeded_fp = os.path.join(data_path, "audio_hex-pickup_debleeded", filename + "_hex_cln.wav")
        hex_y, sr = torchaudio.load(hex_debleeded_fp)
        assert sr == self.sample_rate, f"Sample rate mismatch: {sr} != {self.sample_rate}"
        # crop to a multiple of sample_rate
        hex_y = hex_y[:, :hex_y.shape[1] - (hex_y.shape[1] % self.sample_rate)]
        sample["hex_audio"] = hex_y
        n_channels = hex_y.shape[0]
        assert n_channels == 6, f"Expected 6 channels, got {n_channels}"
        hex_clip_seconds = hex_y.shape[1] / self.sample_rate

        sample["clip_seconds"]=hex_clip_seconds
        n_feature_frames= int(hex_clip_seconds * self.feature_frame_rate)

        # compute features
        pitch, periodicity = compute_pitch(
            hex_y, 
            sample_rate=self.sample_rate,
            fmin=GUITAR_F0_MIN_HZ,
            fmax=GUITAR_F0_MAX_HZ,
            hop_frames=hop_frames,
            device=self.pitch_extraction_device,
            batch_size=self.pitch_extraction_batch_size,
            pad=True
        )
        sample["hex_pitch"] = resample_feature(pitch, n_feature_frames, mode="linear")
        sample["hex_periodicity"] = resample_feature(periodicity, n_feature_frames, mode="linear")

        rms = compute_rms(hex_y, window_size=loudness_window_frames, hop_frames=hop_frames)
        sample["hex_rms"] = resample_feature(rms, n_feature_frames, mode="linear")

        centroid = compute_spectral_centroid(hex_y, win_length=loudness_window_frames, hop_length=hop_frames, sample_rate=self.sample_rate)
        sample["hex_centroid"] = resample_feature(centroid, n_feature_frames, mode="linear")

        loudness = compute_loudness(hex_y, sample_rate=self.sample_rate, hop_length=hop_frames, n_fft=loudness_window_frames)
        sample["hex_loudness"] = resample_feature(loudness, n_feature_frames, mode="linear")

        # mic
        mic_fp = os.path.join(data_path, "audio_mono-mic", filename + "_mic.wav")
        mic_y, sr = torchaudio.load(mic_fp)
        assert sr == self.sample_rate, f"Sample rate mismatch: {sr} != {self.sample_rate}"
        # crop to a multiple of sample_rate
        mic_y = mic_y[:, :mic_y.shape[1] - (mic_y.shape[1] % self.sample_rate)]
        sample["mic_audio"] = mic_y
        mic_clip_seconds = mic_y.shape[1] / self.sample_rate

        # load annotation
        annotation_fp = os.path.join(data_path, "annotation", filename + ".jams")
        annotation = jams.load(annotation_fp)
        midi_annotations = annotation.search(namespace="note_midi")
        midi_pitch = torch.zeros((n_channels, n_feature_frames))
        midi_activity = torch.zeros((n_channels, n_feature_frames))
        midi_onsets = torch.zeros((n_channels, n_feature_frames))
        midi_offsets = torch.zeros((n_channels, n_feature_frames))
        midi_duration_since_previous_onset = torch.ones((n_channels, n_feature_frames))
        for midi_annotation in midi_annotations:
            string_idx = int(midi_annotation["annotation_metadata"]["data_source"])
            for interval in midi_annotation.data:
                start_frame = int(interval.time * self.feature_frame_rate)
                end_frame = int((interval.time + interval.duration) * self.feature_frame_rate)
                midi_pitch[string_idx, start_frame:end_frame] = interval.value
                midi_activity[string_idx, start_frame:end_frame] = 1
                if start_frame < n_feature_frames:
                    midi_onsets[string_idx, start_frame] = 1
                if end_frame < n_feature_frames:
                    midi_offsets[string_idx, end_frame] = 1
        
        # compute duration since previous onset
        for string_idx in range(n_channels):
            for frame_idx in range(1, n_feature_frames):
                if midi_onsets[string_idx, frame_idx] == 1:
                    midi_duration_since_previous_onset[string_idx, frame_idx] = 0
                else:
                    # up to 60 seconds 
                    midi_duration_since_previous_onset[string_idx, frame_idx] = midi_duration_since_previous_onset[string_idx, frame_idx-1] + 1/float(self.feature_frame_rate * 60)

        sample["midi_pitch"] = midi_pitch
        sample["midi_activity"] = midi_activity
        sample["midi_onsets"] = midi_onsets
        sample["midi_offsets"] = midi_offsets
        sample["midi_duration_since_previous_onset"] = midi_duration_since_previous_onset
        return sample

    def set_requested_features(self, requested_features):
        self.requested_features = requested_features

    def fill_midi_pitch(self):
        '''
        Replace 0s in midi pitch replacing with subsequent non-zero values.
        Last non-zero value is repeated until the end of the clip.
        All zero strings are replaced with midi pitch value of open string.
        '''
        print("forward filling midi pitch...")

        for i in tqdm(range(len(self.data))):
            self.data[i]["midi_pitch"] = forward_fill_midi_pitch(self.data[i]["midi_pitch"])

    def compute_pseudo_velocity(self):
        print("computing pseudo midi velocity...")
        for i in tqdm(range(len(self.data))):
                self.data[i]["midi_pseudo_velocity"] = util.scale_db(torch.stack([compute_pseudo_velocity(self.data[i]["midi_activity"][string_index], self.data[i]["hex_loudness"][string_index]) for string_index in range(6)]))
    
    
    def resample_audio(self, new_sample_rate):
        if new_sample_rate != self.sample_rate:
            print("resampling audio...")
            for i in tqdm(range(len(self.data))):
                for feature in self.data[i]:
                    if feature.endswith("audio"):
                        self.data[i][feature] = torchaudio.transforms.Resample(self.sample_rate, new_sample_rate)(self.data[i][feature])
            self.sample_rate = new_sample_rate
    
    def resample_features(self, new_feature_frame_rate):
        if new_feature_frame_rate != self.feature_frame_rate:
            print("resampling features...")
            for i in tqdm(range(len(self.data))):
                for feature in self.data[i]:
                    if "midi" in feature:
                        # do discrete resampling. keeping majority class
                        self.data[i][feature] = resample_feature(self.data[i][feature], int(self.data[i][feature].shape[-1] * new_feature_frame_rate / self.feature_frame_rate), mode="nearest")

                    else:
                        if feature.endswith("pitch") or feature.endswith("periodicity") or feature.endswith("rms") or feature.endswith("centroid") or feature.endswith("loudness") or feature.endswith("velocity") or feature.endswith("activity"):
                            self.data[i][feature] = resample_feature(self.data[i][feature], int(self.data[i][feature].shape[-1] * new_feature_frame_rate / self.feature_frame_rate), mode="linear")
            self.feature_frame_rate = new_feature_frame_rate

    def median_filter_pitch(self, window_size):
        print("median filtering pitch...")
        for i in tqdm(range(len(self.data))):
                self.data[i]["hex_pitch"] = median_filtering(self.data[i]["hex_pitch"], window_size)

    def repair_spectral_centroid(self):
        print("repairing spectral centroid...")
       
        
        for i in tqdm(range(len(self.data))):
            centroid = self.data[i]["hex_centroid"]
            for j in range(centroid.shape[0]):
                centroid[j] = replace_nan_with_previous(centroid[j])
            self.data[i]["hex_centroid"] = centroid

    def __len__(self):
        return self.len
    
    def reindex_data(self, clip_duration, samples_per_clip="n_seconds"):
        self.requested_clip_duration = clip_duration
        self.sample_index_to_data_index = {}
        self.sample_index_to_clip_offset_seconds = {}
        sample_index = 0
        for i in range(len(self.data)):
            if samples_per_clip == "n_seconds":
                for j in range(int(self.data[i]["clip_seconds"]) - self.requested_clip_duration - 1):
                    self.sample_index_to_data_index[sample_index] = i
                    self.sample_index_to_clip_offset_seconds[sample_index] = j
                    sample_index += 1
        self.len = sample_index

    def reindex_data_full_clips(self):
        self.requested_clip_duration = "full"
        self.sample_index_to_data_index = {}
        self.sample_index_to_clip_offset_seconds = {}
        sample_index = 0
        for i in range(len(self.data)):
            self.sample_index_to_data_index[sample_index] = i
            self.sample_index_to_clip_offset_seconds[sample_index] = 0
            sample_index += 1
        self.len = sample_index

    def __getitem__(self, sample_idx):

        idx = self.sample_index_to_data_index[sample_idx]

        # get clip_seconds
        clip_seconds = self.data[idx]["clip_seconds"]
        # convert to feature frames
        clip_frames = int(clip_seconds * self.feature_frame_rate)

        if self.requested_clip_duration == "full":
            start_sample = 0
            end_sample = self.data[idx]["hex_audio"].shape[1]
            start_frame = 0
            end_frame = clip_frames
        else:
            start_seconds = self.sample_index_to_clip_offset_seconds[sample_idx]
            if self.use_random_offset:
                random_offset_seconds = np.random.randint(0, 1)
                start_seconds += random_offset_seconds
            
            start_frame = int(start_seconds * self.feature_frame_rate)
        
            # convert to frames
            end_frame = int(start_frame + self.requested_clip_duration * self.feature_frame_rate)

            start_sample = int(start_frame * self.sample_rate / self.feature_frame_rate)
            end_sample = int(end_frame * self.sample_rate / self.feature_frame_rate)
  
        sample={}

        string_indices = np.array([0, 1, 2, 3, 4, 5])
        sample["string_index"] = einops.rearrange(string_indices, "c -> c 1")

        if "hex_centroid_scaled" in self.requested_features:
            self.requested_features.append("hex_centroid")
            # remove "hex centroid scaled" from requested features
            self.requested_features = [feature for feature in self.requested_features if feature != "hex_centroid_scaled"]

        if self.requested_features == "all":
            return self.data[idx]
        else:
            for feature in self.requested_features:          
                if feature in ["hex_audio", "mic_audio"]:
                    cropped_feature = self.data[idx][feature][:, start_sample:end_sample]
                    sample[feature] = cropped_feature
                elif feature in ["hex_pitch", "hex_periodicity", "hex_rms", "hex_centroid", "hex_loudness", "midi_pitch","midi_activity","midi_pseudo_velocity", "midi_onsets", "midi_offsets", "midi_duration_since_previous_onset"]:
                    cropped_feature = self.data[idx][feature][:, start_frame:end_frame]
                    sample[feature] = cropped_feature
                elif feature == "string_index":
                    continue
                else:
                    sample[feature] = self.data[idx][feature]

        if "hex_centroid" in self.requested_features:
            sample["hex_centroid_scaled"] = util.hz_to_unit(sample["hex_centroid"], 20, self.sample_rate//2,clip=True)

        sample = {**sample, "bpm":self.data[idx]["bpm"], "filename":self.data[idx]["filename"], "n_beats": 48 if self.data[idx]["progression_id"] == "1" else 64, "solo_or_comp": self.data[idx]["solo_or_comp"], "key": self.data[idx]["key"]}

        return sample
        
    def load_data(self,path):
        self.__dict__ = torch.load(path)

    def save_data(self,path):
        torch.save(self.__dict__,path)

if __name__ == "__main__":

    SAMPLE_RATE = 44100
    PITCH_EXTRACTION_BATCH_SIZE = 2000
    FEATURE_FRAME_RATE = 245

    pitch_extraction_device = "cuda:7"

    val_filepaths_path = "./splits/val_filenames.txt"
    val_filenames = []
    with open(val_filepaths_path, "r") as f:
        for line in f:
            val_filenames.append(line.strip())
    val_filepaths = ["./data/GuitarSet/annotation/" + filename for filename in val_filenames]
    val_ds = GuitarSetDataset(val_filepaths, None,sample_rate=SAMPLE_RATE, pitch_extraction_device=pitch_extraction_device, pitch_extraction_batch_size=PITCH_EXTRACTION_BATCH_SIZE, feature_frame_rate=FEATURE_FRAME_RATE)
    val_ds.save_data("./artefacts/guitarset_dataset_data_val.pt")

    tst_filepaths_path = "./splits/tst_filenames.txt"
    tst_filenames = []
    with open(tst_filepaths_path, "r") as f:
        for line in f:
            tst_filenames.append(line.strip())
    tst_filepaths = ["./data/GuitarSet/annotation/" + filename for filename in tst_filenames]
    tst_ds = GuitarSetDataset(tst_filepaths, None,sample_rate=SAMPLE_RATE, pitch_extraction_device=pitch_extraction_device, pitch_extraction_batch_size=PITCH_EXTRACTION_BATCH_SIZE, feature_frame_rate=FEATURE_FRAME_RATE)
    tst_ds.save_data("./artefacts/guitarset_dataset_data_tst.pt")

    trn_filepaths_path = "./splits/trn_filenames.txt"
    trn_filenames = []
    with open(trn_filepaths_path, "r") as f:
        for line in f:
            trn_filenames.append(line.strip())
    trn_filepaths = ["./data/GuitarSet/annotation/" + filename for filename in trn_filenames]
    trn_ds = GuitarSetDataset(trn_filepaths, None,sample_rate=SAMPLE_RATE, pitch_extraction_device=pitch_extraction_device, pitch_extraction_batch_size=PITCH_EXTRACTION_BATCH_SIZE, feature_frame_rate=FEATURE_FRAME_RATE)
    trn_ds.save_data("./artefacts/guitarset_dataset_data_trn.pt")
