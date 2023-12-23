import json
import math

import einops
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import loss
import control_model
import data
import util
import wandb
from data import GuitarSetDataset
from preprocessing import preprocess_model_inputs
from train_synthesis import SynthesisBase
from util import convert_dtype, linear_dequantize, linear_quantize, weights_init, Quantizer
import synthesis_model


class ControlBase(pl.LightningModule):
    def __init__(self, config, dataloader=None):
        super().__init__()

        self.config = config 

        if "with_z" not in self.config:
            self.config["with_z"] = False

        if "pitch_loss_weight" not in self.config:
            self.config["pitch_loss_weight"] = 1.0

        for feature in self.config["regression_features"]:
            if "n_features" not in feature:
                feature["n_features"] = 1

        if "end2end" not in self.config:
            self.config["end2end"] = False


        if "reinitialize_synthesis_model" not in self.config:
            self.config["reinitialize_synthesis_model"] = False

        if "train_synthesis" not in self.config:
            self.config["train_synthesis"] = False

        if "use_spectral_loss" not in self.config:
            self.config["use_spectral_loss"] = False

        if "add_midi_pitch" not in self.config:
            self.config["add_midi_pitch"] = False
        if "add_midi_activity" not in self.config:
            self.config["add_midi_activity"] = False

        if "big_skip_connection" not in self.config:
            self.config["big_skip_connection"] = False
            
        if self.config["quantization_type"] == "linear_minmax":
            if dataloader is not None:
                self.fit_quantizers(dataloader)
        elif self.config["quantization_type"] == "linear_range":
            self.fit_quantizers()

        if self.config["end2end"]:
            N_HARMONICS = self.config["n_harmonics"] if "n_harmonics" in self.config else 128
            N_NOISE_BANDS = self.config["n_noise_bands"] if "n_noise_bands" in self.config else 128
            IR_DURATION = self.config["ir_duration"] if "ir_duration" in self.config else 0.25
            if len(self.config["classification_features"]) > 0:
                self.config["regression_features"] = []
            self.config["regression_features"] = self.config["regression_features"]+[
                {"name":"harmonic_partial_amp_output",
                    "n_features": N_HARMONICS,
                },
                {"name":"harmonic_global_amp_output",
                    "n_features": 1,
                },
                {"name":"noise_band_amp_output",
                    "n_features": N_NOISE_BANDS,
                }
                    ]
            self.ddsp_model = synthesis_model.DDSPModel(
                self.config["model_sample_rate"],
                N_HARMONICS,
                N_NOISE_BANDS,
                ir_duration=IR_DURATION,
                use_one_ir_per_voice=True,
                input_ft_splits=(),
                get_decoder=lambda a,b: None,
                min_f0_hz=data.GUITAR_F0_MIN_HZ,
                max_f0_hz=data.GUITAR_F0_MAX_HZ,
                voice_embedding_size=config["hidden_size"],
                noise_bias=-3.0,
                n_voices=6
            )

        else:

            if self.config["with_z"]:

                self.config["regression_features"] = [{"name":f"z{i}", "n_features":1} for i in range(self.config["hidden_size"])]
                
                # load 
                new_config = {
                'n_voices': 6,
                'seed': 0,
                'loss_fft_sizes': self.config["loss_fft_sizes"],
                # run config
                'model_sample_rate': self.config["model_sample_rate"],
                'model_ft_frame_rate': self.config["model_ft_frame_rate"],
                'n_seconds': self.config["n_seconds"],
                'pitch_median_filter_window_size':self.config["pitch_median_filter_window_size"],
                # synth model config
                'ir_duration': 0.25,
                'n_harmonics': 128,
                'n_noise_bands': 128,
                'one_ir_per_voice': True,
                'acoustic_features':[ "hex_f0_scaled"] + [ft["name"] for ft in self.config["regression_features"]],
                'n_stacks': 6,
                'description': "lstm 128 3 stacks, 64 fps",
                "architecture": "lstm",
                'hidden_size': self.config["hidden_size"],
                "voice_embedding_size": self.config["hidden_size"],
                "rnn_layers":3,
                "noise_bias":-3,
                "n_samples":self.config["n_samples"],
                "learning_rate":0,
                "learning_rate_gamma":0,
                }
                self.synthesis_model = SynthesisBase(new_config)
            else:
                self.synthesis_model = SynthesisBase.load_from_checkpoint(config["synthesis_model_checkpoint"])

                if self.config["reinitialize_synthesis_model"]:
                    # reinitialize synthesis model weights
                    print("reinitializing synthesis model weights")
                    self.synthesis_model.apply(weights_init)

                if not self.config["train_synthesis"]:
                    for param in self.synthesis_model.parameters():
                        param.requires_grad = False

        # print(f"regression features: {self.config['regression_features']}")

        self.learning_rate = config["learning_rate"]
        self.learning_rate_gamma = config["learning_rate_gamma"]

        self.string_embedding_layer = torch.nn.Embedding(self.config["n_voices"], self.config["hidden_size"])

        if len(self.config["continuous_inputs"]) > 0: 
            self.input_size = self.config["hidden_size"] + len(self.config["continuous_inputs"])

            self.input_block = torch.nn.Sequential(
                torch.nn.Linear(self.input_size, self.config["hidden_size"]),
                torch.nn.ReLU(),
                torch.nn.Linear(self.config["hidden_size"], self.config["hidden_size"]),
                torch.nn.ReLU(),
                torch.nn.Linear(self.config["hidden_size"], self.config["hidden_size"]),
                torch.nn.ReLU()
            )
        
        self.input_embedding_layers = torch.nn.ModuleDict()
        self.input_quantizers={}
        for feature in self.config["discrete_inputs"]:
            feature_name = feature["name"]
            self.input_quantizers[feature_name] = Quantizer(feature["range"], feature["n_bins"])
            self.input_embedding_layers[feature_name] = torch.nn.Embedding(feature["n_bins"], self.config["hidden_size"])

        if self.config["architecture"] == "cnn":
            self.main_block =torch.nn.Sequential(
                *[control_model.SACNNBlock(input_channels=self.config["n_voices"],
                            hidden_size=self.config["hidden_size"],
                            num_heads=self.config["n_heads"],
                            norm_type=self.config["norm_type"],
                            kernel_size=2,
                            dilations=[1,2,4,8,16,32,64,128],
                            activation="gated"
                            )
                for _ in range(self.config["n_blocks"])])
        elif self.config["architecture"] == "transformer_encoder":
            self.main_block =control_model.TransformerEncoderModel(
                n_layers=self.config["n_blocks"],
                n_heads=self.config["n_heads"],
                hidden_size=self.config["hidden_size"],
            )
        elif self.config["architecture"] == "transformer":
            self.main_block = control_model.FullTransformerModel(
                n_layers=self.config["n_blocks"],
                n_heads=self.config["n_heads"],
                hidden_size=self.config["hidden_size"],
            )
        else:
            self.main_block =torch.nn.Sequential(
                    *[control_model.SARNNBlock(input_channels=self.config["n_voices"],
                                hidden_size=self.config["hidden_size"],
                                num_heads=self.config["n_heads"],
                                n_rnn_layers_per_block=self.config["n_rnn_layers_per_block"],
                                rnn_type=self.config["architecture"],
                                norm_type=self.config["norm_type"])
                    for _ in range(self.config["n_blocks"])])

            self.total_classification_output_size = sum(feature["n_bins"] for feature in self.config["classification_features"])
            self.total_regression_output_size = sum(feature["n_features"] for feature in self.config["regression_features"])
            self.n_output_size = self.total_classification_output_size + self.total_regression_output_size
            self.output_block = torch.nn.Sequential(
            torch.nn.Linear(self.config["hidden_size"], self.n_output_size)
            )
            # count number of parameters in output block
                # n_params = 0
                # for param in self.output_block.parameters():
                #     n_params += param.numel()
                # print(f"output block has {n_params} parameters")
           
        self.save_hyperparameters()

    def encode(self, inputs):
        batch, channel, t, _ = inputs["midi_pitch_scaled"].shape

        string_index = torch.arange(0, channel, device=self.device)
        string_z = self.string_embedding_layer(string_index)
        input_z = einops.repeat(string_z, 'c ft-> b c t ft', b=batch, t=t, c=channel)

        for feature in self.config["discrete_inputs"]:
            feature_name = feature["name"]
            feature_values = inputs[feature_name]

            # print quantizater max and min

            
            feature_values = self.input_quantizers[feature_name].quantize(feature_values)
            

            feature_z = self.input_embedding_layers[feature_name](feature_values)
            input_z = input_z + feature_z 

        for input_feature in self.config["continuous_inputs"]:
            feature_name = input_feature["name"]
            feature_values = inputs[feature_name]
            input_z = torch.cat([input_z, feature_values], dim=-1)

        if len(self.config["continuous_inputs"]) > 0:
            input_z = self.input_block(input_z)

        return input_z
    
    def decode(self, out, inputs):
        output = {}

        out = self.output_block(out)

        classification_outputs, regression_outputs = torch.split(out, [self.total_classification_output_size, self.total_regression_output_size], dim=-1)
        regression_outputs = torch.split(regression_outputs, [feature["n_features"] for feature in self.config["regression_features"]], dim=-1)

        classification_outputs = torch.split(classification_outputs, [feature["n_bins"] for feature in self.config["classification_features"]], dim=-1)
        
        for i, feature in enumerate(self.config["classification_features"]):
            feature_name = feature["name"]
            output[feature_name+"_logits"] = classification_outputs[i]
            if feature_name == "hex_f0_scaled" and self.config["add_midi_pitch"]:
                midi_pitch_scaled = self.quantizers[feature_name].quantize(inputs["midi_pitch_scaled"])
                midi_pitch_scaled = torch.nn.functional.one_hot(midi_pitch_scaled, self.quantizers[feature_name].n_bins).to(inputs["midi_pitch_scaled"].device).to(inputs["midi_pitch_scaled"].dtype)
                output["hex_f0_scaled_logits"] = output["hex_f0_scaled_logits"] + midi_pitch_scaled
            # if feature_name == "hex_loudness_scaled" and self.config["add_midi_activity"]:
            #     output["hex_loudness_scaled_logits"] = output["hex_loudness_scaled_logits"] + self.quantizers[feature_name].quantize(inputs["midi_activity"])
            output[feature_name+"_hat"] = self.quantizers[feature_name].dequantize(torch.argmax(output[feature_name+"_logits"], dim=-1))[...,None]
        
        for i, feature in enumerate(self.config["regression_features"]):
            feature_name = feature["name"]
            if feature_name == "hex_f0_scaled":
                output["hex_f0_scaled_hat"] = torch.sigmoid(regression_outputs[i])
            elif feature_name == "hex_loudness_scaled":
                output["hex_loudness_scaled_hat"] = torch.nn.functional.softplus(regression_outputs[i])
            elif feature_name == "hex_periodicity":
                output["hex_periodicity_hat"] = torch.sigmoid(regression_outputs[i])
            elif feature_name == "hex_centroid_scaled":
                output["hex_centroid_scaled_hat"] = torch.sigmoid(regression_outputs[i])
            elif feature_name == "harmonic_partial_amp_output":
                output["harmonic_partial_amp_output"] = regression_outputs[i]
            elif feature_name == "harmonic_global_amp_output":
                output["harmonic_global_amp_output"] = regression_outputs[i]
            elif feature_name == "noise_band_amp_output":
                output["noise_band_amp_output"] = regression_outputs[i]
            # if feature name is zn where n is an integer
            elif "z" in feature_name[:1]:
                output[feature_name] = regression_outputs[i]
        return output
     
    def fit_quantizers(self, dataloader=None):
        self.quantizers={}

        if self.config["quantization_type"] == "linear_minmax":
            feature_records = {}
            for sample in tqdm(dataloader):
                sample = preprocess_model_inputs(sample)
                # feature record
                for feature in self.config["classification_features"]:
                    feature_name = feature["name"]
                    if feature_name not in feature_records:
                        feature_records[feature_name] = []
                    else:
                        feature_records[feature_name].append(sample[feature_name].cpu().flatten().numpy())

            for feature in self.config["classification_features"]:
                feature_name = feature["name"]
                feature_records[feature_name] = np.concatenate(feature_records[feature_name])
                print(f"{feature} min: {np.min(feature_records[feature_name])}, max: {np.max(feature_records[feature_name])}")
                feature_name = feature["name"]
                n_bins = feature["n_bins"]
                print(f"fitting quantizer for {feature_name} with {n_bins} bins")
                # sort values
                feature_record = feature_records[feature_name]
                self.quantizers[feature_name] = Quantizer(feature_record, n_bins)

        elif self.config["quantization_type"] == "linear_range":
            for feature in self.config["classification_features"]:
                feature_name = feature["name"]
                n_bins = feature["n_bins"]
                self.quantizers[feature_name] = Quantizer(feature["range"], feature["n_bins"])

    def forward(self, x):
        inputs = x
        x1 = self.encode(x)
        x2 = self.main_block(x1)
        if self.config["big_skip_connection"]:
            x2 = x2 + x1
        x3 = self.decode(x2,inputs)
        if "voice_index" in inputs:
            x3 = {**x3, "voice_index":inputs["voice_index"]}
        return x3

    def classification_loss(self,target_onehot,prediction_logits):
        b,v,t,f = prediction_logits.shape
        target_onehot = einops.rearrange(target_onehot, "b v t f -> (b v t) f")
        prediction_logits = einops.rearrange(prediction_logits, "b v t n -> (b v t) n")

        # print shape and dtype
        ce_loss = torch.nn.functional.cross_entropy(prediction_logits, target_onehot, reduction="none")
        # reshape to (batch_size, n_voices, time)
        loss = einops.rearrange(ce_loss, "(b v t) -> b v t 1", b=b, v=v, t=t)
        return loss
      
    def loss(self, targets, predictions):
        losses = {}
        
        
        for i, feature in enumerate(self.config["classification_features"]):
            feature_name = feature["name"]
            batch, channel, time, fts = targets[feature_name].shape
            # create high resolution target distribution
            targets_class = self.quantizers[feature_name].quantize(targets[feature_name])
            target_onehot = torch.nn.functional.one_hot(targets_class, feature["n_bins"]).to(targets[feature_name].device).to(targets[feature_name].dtype)
            # now add gaussian blur with standard deviation of 0.01 of the total range
            if self.config["class_smoothing_sigma"] > 0:
                # make kernel
                kernel = torch.arange(-feature["n_bins"]//2, feature["n_bins"]//2+1, dtype=targets[feature_name].dtype, device=targets[feature_name].device)
                sigma = self.config["class_smoothing_sigma"] * feature["n_bins"]
                kernel = torch.exp(-kernel**2/(2*sigma))
                kernel = kernel / torch.sum(kernel)
                kernel = einops.rearrange(kernel, "n -> () () n")
                target_onehot = einops.rearrange(target_onehot, "b v t f -> (b v t) 1 f")
                target_onehot = torch.nn.functional.conv1d(target_onehot, kernel, padding="same")
                target_onehot = einops.rearrange(target_onehot, "(b v t) 1 f -> b v t f", b=batch, v=channel, t=time)
         
            closs = self.classification_loss(
                target_onehot,
                predictions[feature_name+"_logits"],
            )
            losses[feature_name+"_loss_contour"] = closs
        
        for i, feature in enumerate(self.config["regression_features"]):
            feature_name = feature["name"]
            #if not self.config["use_spectral_loss"] and feature_name != "hex_f0_scaled" and feature_name in targets:
            if feature_name in targets:
                rloss = (targets[feature_name] - predictions[feature_name+"_hat"])**2
                # print(f"{feature_name} rloss shape: {rloss.shape}")
                losses[feature_name+"_loss_contour"] = rloss

                # print(f"{feature_name} shape: {rloss.shape}")

        rescaled_loudness = targets["hex_loudness_scaled"] - 1
        periodicity = targets["hex_periodicity"]

        losses["pitch_loss"] = torch.mean(losses["hex_f0_scaled_loss_contour"] * periodicity * rescaled_loudness)

        losses["loss"] = 0

        losses["loss"] += losses["pitch_loss"]  * self.config["pitch_loss_weight"]

        if not self.config["use_spectral_loss"]:
            if "hex_periodicity_loss_contour" in losses:
                losses["periodicity_loss"] = torch.mean(losses["hex_periodicity_loss_contour"]*rescaled_loudness)
                losses["loss"] +=  losses["periodicity_loss"]
            if "hex_centroid_scaled_loss_contour" in losses:
                losses["centroid_loss"] = torch.mean(losses["hex_centroid_scaled_loss_contour"] * rescaled_loudness)
                losses["loss"] +=  losses["centroid_loss"]
            if "hex_loudness_scaled_loss_contour" in losses:
                losses["loudness_loss"] = torch.mean(losses["hex_loudness_scaled_loss_contour"])
                losses["loss"] +=  losses["loudness_loss"]

        # losses["periodicity_loss"] = torch.mean(losses["hex_periodicity_loss_contour"])
        # losses["centroid_loss"] = torch.mean(losses["hex_centroid_scaled_loss_contour"])
        # losses["loudness_loss"] = torch.mean(losses["hex_loudness_scaled_loss_contour"])


        if self.config["use_spectral_loss"]:

            if self.config["end2end"]:
                synthesis_output = self.render(predictions)
            else:
                if self.config["with_z"]:
                    synthesis_inputs = {**{f"z{i}":predictions[f"z{i}"] for i in range(self.config["hidden_size"])}, **{"voice_index":predictions["voice_index"], "hex_f0_scaled":predictions["hex_f0_scaled_hat"]}}
                else:
                    synthesis_inputs = {"hex_loudness_scaled":predictions["hex_loudness_scaled_hat"], "hex_f0_scaled":predictions["hex_f0_scaled_hat"], "hex_periodicity":predictions["hex_periodicity_hat"], "hex_centroid_scaled":predictions["hex_centroid_scaled_hat"], "voice_index":targets["voice_index"]}
                synthesis_output = self.synthesis_model.forward(synthesis_inputs) 
            
            spectral_loss = loss.multiscale_spectral_loss(
            targets["mic_audio"].squeeze(1),
            synthesis_output["output"].squeeze(1), 
            scales=self.config["loss_fft_sizes"],
            overlap=0.75)

            losses["spectral_loss"] = spectral_loss

            losses["loss"] += losses["spectral_loss"]

        return losses

    def iter_step(self,batch,batch_idx):
        inputs = preprocess_model_inputs(batch)
        inputs = convert_dtype(inputs,{torch.float32:self.dtype, torch.float64:self.dtype}) 
      
        outputs = self(inputs)
 
        losses = self.loss(inputs,outputs)
        outputs = {**outputs,**losses,**inputs, **batch}
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self.iter_step(batch,batch_idx)
        self.log_dict({f"{k}/trn": v for k, v in outputs.items() if k.endswith("loss")}, on_step=True, on_epoch=True, logger=True)
        if batch_idx == 0:
            self.save_demos(outputs, prefix="trn_")
        return outputs["loss"]

    def validation_step(self, batch, batch_idx):
        outputs = self.iter_step(batch,batch_idx)
        self.log_dict({f"{k}/val": v for k, v in outputs.items() if k.endswith("loss")}, on_step=True, on_epoch=True, logger=True)
        if batch_idx == 0:
            self.save_demos(outputs, prefix="val_")
        return outputs["loss"]

    def configure_optimizers(self):
        # 0.99 decay rate per epoch
        # 3e-4 learning rate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.learning_rate_gamma)
        return [self.optimizer], [self.scheduler]
    
    def on_train_epoch_start(self) -> None:
        self.logger.experiment.log({}, commit = True)

    def midi2control(self, midi):
        inputs = preprocess_model_inputs(midi)
        inputs = convert_dtype(inputs,{torch.float32:self.dtype, torch.float64:self.dtype}) 
        output = self(inputs)
        output = {**inputs, **output}
        predicted_acoustic_features = {}
        for key in ["hex_f0_scaled", "hex_loudness_scaled", "hex_periodicity", "hex_centroid_scaled"]:
            predicted_acoustic_features[key] = output[f"{key}_hat"][:1]
        predicted_acoustic_features["voice_index"] = output["voice_index"][:1]
        return {**output, **predicted_acoustic_features}
    
    def render(self, predictions, n_samples=None):

        synthesis_inputs = {"harmonic_partial_amp_output":predictions["harmonic_partial_amp_output"], "harmonic_global_amp_output":predictions["harmonic_global_amp_output"], "noise_band_amp_output":predictions["noise_band_amp_output"], "voice_index":predictions["voice_index"], "hex_f0_scaled":predictions["hex_f0_scaled_hat"].detach()}
        # for key,value in synthesis_inputs.items():
        #     print(f"{key} shape: {value.shape}")
            
        synthesis_outputs = self.ddsp_model.synthesize(synthesis_inputs, self.config["n_samples"] if n_samples is None else n_samples)
        return synthesis_outputs

    def control2audio(self,inputs):
        model_output = self.synthesis_model.forward(inputs)
        output = {**inputs,  **model_output}
        return output

    def save_demos(self, output, prefix=""):
        with torch.no_grad():
            if self.config["end2end"]:
                
                synthesis_output = self.render(output)["output"].cpu()

                self.logger.experiment.log({f"{prefix}demo": wandb.Audio(synthesis_output.numpy().flatten(), caption="demo", sample_rate=model.config["model_sample_rate"]),
                                            f"{prefix}demo_mic_audio": wandb.Audio(output["mic_audio"].cpu().numpy().flatten(), caption="demo", sample_rate=model.config["model_sample_rate"])
                                            }, commit = False)

            else:
                if self.config["with_z"]:
                    predicted_acoustic_features = {**{f"z{i}":output[f"z{i}"] for i in range(self.config["hidden_size"])}, **{"voice_index":output["voice_index"], "hex_f0_scaled":output["hex_f0_scaled_hat"]}}
                else:
                    predicted_acoustic_features = {}
                    for key in ["hex_f0_scaled", "hex_loudness_scaled", "hex_periodicity", "hex_centroid_scaled"]:
                        predicted_acoustic_features[key] = output[f"{key}_hat"][:1]
                    predicted_acoustic_features["voice_index"] = output["voice_index"][:1]
                output_audio = self.synthesis_model.forward(predicted_acoustic_features)["output"]
                # output flattened audio
                self.logger.experiment.log({f"{prefix}demo": wandb.Audio(output_audio.cpu().numpy().flatten(), caption="demo", sample_rate=model.config["model_sample_rate"]),
                                            f"{prefix}demo_mic_audio": wandb.Audio(output["mic_audio"][:1].cpu().numpy().flatten(), caption="demo", sample_rate=model.config["model_sample_rate"])
                                            }, commit = False)

                    # plot f0, loudness, periodicity, centroid and it's predictions
                for key in ["hex_f0_scaled", "hex_loudness_scaled", "hex_periodicity", "hex_centroid_scaled"]:
                    if key+"_hat" in output:
                        self.logger.experiment.log({
                            f"{prefix}{key}": wandb.plot.line_series(
                            xs=range(output[key][:1].flatten().cpu().detach().numpy().shape[0]),
                            ys=[output[key][:1].flatten().cpu().detach().numpy(), output[f"{key}_hat"][:1].flatten().cpu().detach().numpy()],
                            keys=["target", "prediction"],
                            title=f"{prefix}, {key}",
                            xname="time",
                        )}, 
                        commit = False)

                # plot midi pitch
                self.logger.experiment.log(
                    {
                        f"{prefix}midi_pitch_scaled": wandb.plot.line_series(
                            xs=range(output["midi_pitch_scaled"][:1].flatten().cpu().detach().numpy().shape[0]),
                            ys=[output["midi_pitch_scaled"][:1].flatten().cpu().detach().numpy()],
                            keys=["midi_pitch_scaled"],
                            title=f"{prefix}, midi_pitch_scaled",
                            xname="time",
                        )
                    },
                    commit = False,
                )

                # plot midi pseudo velocity
                self.logger.experiment.log(
                    {
                        f"{prefix}pseudo_velocity": wandb.plot.line_series(
                            xs=range(output["midi_pseudo_velocity"][:1].flatten().cpu().detach().numpy().shape[0]),
                            ys=[output["midi_pseudo_velocity"][:1].flatten().cpu().detach().numpy()],
                            keys=["midi_pseudo_velocity"],
                            title=f"{prefix}, midi_pseudo_velocity",
                            xname="time",
                        )
                    },
                    commit = False,
                )

N_PITCHES = math.floor(util.hz_to_midi(data.GUITAR_F0_MAX_HZ))-math.floor(util.hz_to_midi(data.GUITAR_F0_MIN_HZ))
N_BINS_PER_PITCH = 5

wandb_config = {
'n_voices': 6,
'seed': 0,
'batch_size': 2,
'trn_split_ratio': 0.9,
'half_precision': False,
'model_sample_rate': 48000,
'model_ft_frame_rate': 128,
'n_seconds': 8,
'pitch_median_filter_window_size':1,
'synthesis_model_checkpoint': 'artefacts/synthesis_checkpoints/proud-donkey-161.ckpt',
"learning_rate_gamma": 0.99,
"discrete_inputs": [
    {"name": "midi_pitch_scaled",
    "n_bins": N_PITCHES * N_BINS_PER_PITCH,
    "range": [0,1],
}
,
{
    "name":"midi_pseudo_velocity",
    "n_bins": 64,
    "range": [1.0,1.3],       
}
,
# {"name":"midi_onsets", "n_bins":2, "range": [0,1]},
# {"name":"midi_offsets", "n_bins":2, "range": [0,1]},
],
"continuous_inputs": [
#     {"name": "midi_pitch_scaled",
#     "range": [0,1],
# }
# ,
# {"name":"midi_activity",
#     "range": [0,1],
#  },
# {
#     "name":"midi_pseudo_velocity",
#     "range": [1.0,1.3],
# },
# {"name":"midi_duration_since_previous_onset",
#     "range": [0,1],
#  },
],
"classification_features": [
    {"name": "hex_f0_scaled",
    "n_bins": N_PITCHES * N_BINS_PER_PITCH,
    "range": [0,1],
    }
    #, 
# {"name":"hex_loudness_scaled",
#     "n_bins": 64,
#     "range": [1.0,1.3],
# },
# {"name":"hex_periodicity",
#     "n_bins": 64,
#     "range": [0,1],
# },
# {"name":"hex_centroid_scaled",
#     "n_bins": 64,
#     "range": [0,1],
# }
],
'16k_loss_fft_sizes': [64, 128, 256, 512, 1024, 2048, 4096],
"regression_features": [
    # {"name": "hex_f0_scaled",
    # "n_features": 1,
    # }
# , 
{"name":"hex_loudness_scaled",
    "n_features": 1,
},
{"name":"hex_periodicity",
    "n_features": 1,
},
{"name":"hex_centroid_scaled",
    "n_features": 1,
}
    ],
"use_spectral_loss": True,
"architecture": "lstm",
"hidden_size":512,
"n_blocks":3,
"n_heads":1,
"n_rnn_layers_per_block":3,
"norm_type": None,
'gpu': 7,
"learning_rate": 1e-4,
"quantization_type": "linear_range",
"structured_output": False,
"class_smoothing_sigma": 0.00,
"big_skip_connection": False,
"description": "train_synthesis",
"add_midi_pitch": False,
"add_midi_activity": False,
"train_synthesis": True,
"reinitialize_synthesis_model": True,
"end2end":False,
"pitch_loss_weight": 1.0,
"n_harmonics": 128,
"n_noise_bands": 128,
"ir_duration": 0.25,
"with_z": False,
"overfitting_test": False,
"dry_run": False,
}

if __name__ == "__main__":

    checkpoint_path = None #"artefacts/control_checkpoints/peach-mountain-807epoch=8.ckpt"

    if checkpoint_path is not None:
        # load config from checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        
        
        wandb_config = {"gpu": wandb_config["gpu"],
                        **checkpoint["hyper_parameters"]["config"]
        }

    
    # description = input("Enter a description of this run: ")
    wandb_config["loss_fft_sizes"] = [base_fft_size * wandb_config["model_sample_rate"] // 16000 for base_fft_size in wandb_config["16k_loss_fft_sizes"]]

    wandb_config = {**wandb_config,
    'batch_size': 1 if wandb_config['overfitting_test'] or wandb_config['dry_run'] else wandb_config['batch_size'],
    }

    print("WANDB CONFIG")
    print(json.dumps(wandb_config, indent=4))
    print("")

    # init wandb
    wandb.init(project="neural guitar, control", config=wandb_config)

    torch.manual_seed(wandb.config.seed)
    np.random.seed(wandb.config.seed)
        
    if not wandb_config["overfitting_test"]:
        trn_ds = data.load_prepared_data(prepared_data_path="./artefacts/guitarset_dataset_data_trn.pt", seconds_per_clip=wandb.config.n_seconds, sample_rate = wandb.config.model_sample_rate, feature_frame_rate=wandb.config.model_ft_frame_rate, use_random_offset=True)
    else:
        trn_ds = data.load_prepared_data(prepared_data_path="./artefacts/guitarset_dataset_data_val.pt", seconds_per_clip=wandb.config.n_seconds, sample_rate = wandb.config.model_sample_rate, feature_frame_rate=wandb.config.model_ft_frame_rate, use_random_offset=True)
    val_ds = data.load_prepared_data(prepared_data_path="./artefacts/guitarset_dataset_data_val.pt", seconds_per_clip=wandb.config.n_seconds, sample_rate = wandb.config.model_sample_rate, feature_frame_rate=wandb.config.model_ft_frame_rate, use_random_offset=False)
    

    #%%
    example = trn_ds[0]
    n_samples = example["hex_audio"].shape[-1]

    wandb.config.n_samples = n_samples

    wandb_logger = WandbLogger(project="neural-guitar-synthesis", log_model=True, notes=wandb.config.description)
    wandb.run.log_code(".")

    trn_dl = DataLoader(trn_ds, batch_size=wandb.config.batch_size, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=wandb.config.batch_size, shuffle=False, drop_last=False)
    
    model = ControlBase(wandb.config.as_dict(),trn_dl)

    checkpoint_callback = ModelCheckpoint(
        monitor='loss/val_epoch',
        dirpath='./artefacts/control_checkpoints',
        filename=f"{wandb_logger.experiment.name}"+"{epoch}",
        save_top_k=5,
        mode='min',
    )

    trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=-1,
        accelerator='gpu',
        devices=[wandb.config.gpu], 
        precision=16 if wandb.config.half_precision else 32,
        callbacks=[LearningRateMonitor(logging_interval='step'), checkpoint_callback],
        gradient_clip_val=1.0,          
    )
    print("experiment name:", wandb_logger.experiment.name)

    trainer.fit(model, trn_dl, val_dl, ckpt_path=checkpoint_path)