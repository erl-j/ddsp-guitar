#%%
import torch
import data
from train_control import ControlBase

ckpt = "artefacts/listening_test_checkpoints/lyric-disco-751-epoch-21.ckpt"
model = ControlBase.load_from_checkpoint(ckpt)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

sample_rate = model.config["model_sample_rate"]
ft_frame_rate = model.config["model_ft_frame_rate"]
# inspect
tst_ds = data.load_prepared_data(prepared_data_path="./artefacts/guitarset_dataset_data_tst.pt", seconds_per_clip="full", sample_rate=sample_rate, feature_frame_rate=ft_frame_rate, use_random_offset=False)
dl = torch.utils.data.DataLoader(tst_ds, batch_size=1, shuffle=False, drop_last=False)

sample = next(iter(dl))

#%%
print(sample.keys())


import matplotlib.pyplot as plt
fig, axs= plt.subplots(6, 1, figsize=(20, 10))
for i in range(6):
    axs[i].hist(sample["midi_pseudo_velocity"][0, i].cpu().numpy(), bins=100)
    
fig, axs= plt.subplots(6, 1, figsize=(20, 10))
for i in range(6):
    axs[i].plot(sample["midi_pseudo_velocity"][0, i].cpu().numpy())

min_vel = 1.09
max_vel = 1.12


# %%
