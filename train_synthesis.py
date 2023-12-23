import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split
import data
import loss
import synthesis_model
import util
import wandb
from data import GuitarSetDataset
from preprocessing import preprocess_model_inputs
from util import convert_dtype, resample_feature, weights_init
import json
import control_model

class SynthesisBase(pl.LightningModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.save_hyperparameters()

        if "noise_bias" not in config:
            config["noise_bias"]=-3

        if "architecture" not in config:
            config["architecture"] = "wavenet"

        if "voice_embedding_size" not in config:
            cond_size = 6
            config["voice_embedding_size"] = -1
        else:
            cond_size = config["voice_embedding_size"]

        if config["architecture"] == "gru" or config["architecture"] == "lstm" or config["architecture"] == "rnn":
            def get_decoder(input_splits, output_splits):
                return synthesis_model.MixFcDecoder(
                    rnn_channels=config["hidden_size"],
                    rnn_type=config["architecture"],
                    ch=config["hidden_size"],
                    layers_per_stack=3,
                    rnn_layers=config["rnn_layers"] if "rnn_layers" in config else 1,
                    input_splits=input_splits,
                    output_splits=output_splits
                    )
                
            
        if "wavenet" in config["architecture"]:
            def get_decoder(input_splits, output_splits):
                return synthesis_model.DDSPDecoderAdapter(
                    input_splits=input_splits,
                    output_splits=output_splits,
                    is_wavenet=True,
                    model=synthesis_model.DilatedConvStackStack(
                        input_size=sum([s for _, s in input_splits]),
                        cond_size=cond_size,
                        hidden_size=config["hidden_size"],
                        output_size=sum([s for _, s in output_splits]),
                        dilations=[config["dilations"]] * config["n_stacks"],
                        causal=config["causal"],
                        activation=config["activation"],
                        kernel_size=config["kernel_size"],
                        n_attention_heads=4 if "wavenet_sa" in config["architecture"] else 0,
                    )
                )
        elif config["architecture"] == "sacnn":
            def get_decoder(input_splits, output_splits):
                return synthesis_model.DDSPDecoderAdapter(
                    input_splits=input_splits,
                    output_splits=output_splits,
                    model=control_model.SACNNStack(
                    input_size = sum([s for _, s in input_splits])+config["voice_embedding_size"],
                    output_size = sum([s for _, s in output_splits]),
                    hidden_size=config["hidden_size"],
                    n_blocks=config["n_stacks"],
                    kernel_size=config["kernel_size"],
                    activation=config["activation"],
                    # causal=config["causal"],
                    dilations=config["dilations"],
                    n_channels=config["n_voices"],
                    n_heads=4,
                    norm_type=config["norm_type"] if "norm_type" in config else "layer norm c t f",
                )
            )   
        elif  config["architecture"] == "sarnn":
            def get_decoder(input_splits, output_splits):
                return synthesis_model.DDSPDecoderAdapter(
                    input_splits=input_splits,
                    output_splits=output_splits,
                    model=control_model.SARNNStack(
                    input_size = sum([s for _, s in input_splits])+config["voice_embedding_size"],
                    output_size = sum([s for _, s in output_splits]),
                    hidden_size=config["hidden_size"],
                    n_blocks=config["n_stacks"],
                    # causal=config["causal"],
                    n_channels=config["n_voices"],
                    n_heads=4,
                    norm_type="layer norm c t f",
                    rnn_type="lstm",
                    n_rnn_layers_per_block=1,
                )
            )   

        self.synthesis_model = synthesis_model.DDSPModel(
            sample_rate=config["model_sample_rate"],
            n_harmonics=config["n_harmonics"],
            n_noise_bands=config["n_noise_bands"],
            ir_duration=config["ir_duration"],
            min_f0_hz=data.GUITAR_F0_MIN_HZ,
            max_f0_hz=data.GUITAR_F0_MAX_HZ,
            input_ft_splits=tuple((ft, 1) for ft in config["acoustic_features"]),
            use_one_ir_per_voice=config["one_ir_per_voice"],
            voice_embedding_size=config["voice_embedding_size"],
            n_voices=config["n_voices"],
            get_decoder=get_decoder,
            noise_bias=config["noise_bias"]
        ).to(self.dtype)

        self.sample_rate = config["model_sample_rate"]
        self.n_samples = config["n_samples"]
        self.fft_sizes = config["loss_fft_sizes"]
        self.learning_rate = config["learning_rate"]
        self.learning_rate_gamma = config["learning_rate_gamma"]


    def forward(self, preprocessed_inputs):
        return self.synthesis_model(preprocessed_inputs, self.n_samples)

    def iter_step(self,batch,batch_idx):
        inputs = batch
        inputs = convert_dtype(inputs,{torch.float32:self.dtype, torch.float64:self.dtype})  
        preprocessed_inputs = preprocess_model_inputs(inputs)

        # check for inf and nan in preprocessed inputs
        for k,v in preprocessed_inputs.items():
            if torch.isnan(v).any() or torch.isinf(v).any():
                raise ValueError(f"nan or inf in {k}")

        outputs = self(preprocessed_inputs)
        # check for nan and infs
        if torch.isnan(outputs["output"]).any() or torch.isinf(outputs["output"]).any():
            raise ValueError("nan or inf in output")
        loss_value = loss.multiscale_spectral_loss(
            inputs["mic_audio"].squeeze(1),
            outputs["output"].squeeze(1), 
            scales=self.fft_sizes,
            overlap=0.75)
        outputs["loss"] = loss_value
        outputs ={**outputs,**preprocessed_inputs, **inputs}
        return outputs  

    def training_step(self, batch, batch_idx):
        outputs = self.iter_step(batch,batch_idx)
        self.log("train_loss", outputs["loss"], on_epoch=True, prog_bar=True)
        self.log_dict({f"loss/trn": outputs["loss"]}, on_step=True, on_epoch=True, logger=True)
        if batch_idx == 0:
            self.save_demos(outputs, prefix="train_")
        return outputs["loss"]

    def validation_step(self, batch, batch_idx):
        outputs = self.iter_step(batch,batch_idx)
        self.log("val_loss", outputs["loss"], prog_bar=True,  on_epoch=True)
        self.log_dict({f"loss/val": outputs["loss"]}, on_step=True, on_epoch=True, logger=True)
        # this is a hack to save the first batch of the first epoch
        if batch_idx == 0:
            self.save_demos(outputs, prefix="val_")
        return outputs["loss"]

    def save_demos(self, output, prefix =""):
        # save mic audio 
        self.logger.experiment.log({
            f"{prefix}_mic_audio": [wandb.Audio(output["mic_audio"][0].flatten().detach().cpu().float(), caption=f"{prefix}_mic_audio", sample_rate=self.sample_rate)],
            f"{prefix}_output": [wandb.Audio(output["output"][0].flatten().detach().cpu().float(), caption=f"{prefix}_output", sample_rate=self.sample_rate)],
            f"{prefix}_string_harmonic_output": [wandb.Audio(output["string_harmonic_output"][0].flatten().detach().cpu().float(), caption=f"{prefix}_string_harmonic_output", sample_rate=self.sample_rate)],
            f"{prefix}_string_noise_output": [wandb.Audio(output["string_noise_output"][0].flatten().detach().cpu().float(), caption=f"{prefix}_string_noise_output", sample_rate=self.sample_rate)],
            f"{prefix}_dry_mix": [wandb.Audio(output["dry_mix"][0].flatten().detach().cpu().float(), caption=f"{prefix}_dry_mix", sample_rate=self.sample_rate)],
            f"{prefix}_reverb_mix" :[wandb.Audio(output["reverb_mix"][0].flatten().detach().cpu().float(), caption=f"{prefix}_reverb_mix", sample_rate=self.sample_rate)],
            f"{prefix}_ir":[wandb.Audio(torch.flip(output["ir"],dims=[-1]).flatten().detach().cpu().float(), caption=f"{prefix}_ir", sample_rate=self.sample_rate)],
        }, commit = False )

    def on_train_epoch_start(self) -> None:
        self.logger.experiment.log({}, commit = True)

    def configure_optimizers(self):
        # 0.99 decay rate per epoch
        # 3e-4 learning rate
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.learning_rate_gamma)
        return [self.optimizer], [self.scheduler]

def profile_model(model, ds, batch_size, sample_rate):
    inputs = next(iter(DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)))
    model.to("cuda")
    inputs_dt = convert_dtype(inputs,{torch.float32:model.dtype, torch.float64:model.dtype}) 
    inputs_dt = {k:v.to("cuda") for k,v in inputs_dt.items() if isinstance(v,torch.Tensor)}
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=100, warmup=100, active=100, repeat=10),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        prof.start()
        for i in range(202):
            outputs = model(preprocess_model_inputs(inputs_dt))
            prof.step()

wandb_config = {
    # domain constants
    'n_voices': 6,
    'seed': 0,
    '16k_loss_fft_sizes': [64, 128, 256, 512, 1024, 2048, 4096],
    # run config
    'gpu': 0,
    'dry_run': False,
    'overfitting_test': False,
    'batch_size':3,
    'learning_rate': 3e-4,
    'learning_rate_gamma': 0.99,
    'trn_split_ratio': 0.9,
    'half_precision': False,
    'model_sample_rate': 48000,
    'model_ft_frame_rate': 128,
    'n_seconds': 8,
    'pitch_median_filter_window_size':1,
    # synth model config
    'ir_duration': 0.25,
    'n_harmonics': 128,
    'n_noise_bands': 128,
    'one_ir_per_voice': True,
    'acoustic_features': ["hex_f0_scaled", "hex_loudness_scaled", "hex_periodicity", "hex_centroid_scaled"],
    'n_stacks': 6,
    'dilations': [1, 2, 4, 8, 16],
    'kernel_size': 3,
    'causal': False,
    'activation': "gated",
    'description': "lstm 128 3 stacks, 64 fps",
    "use_cached_preprocessed_dataset": False,
    "architecture": "lstm",
    'hidden_size': 512,
    "voice_embedding_size": 512,
    "norm_type": None,
    "rnn_layers":3,
    "noise_bias":-3,
}   


if __name__ == "__main__":
    wandb_config["loss_fft_sizes"] = [base_fft_size * wandb_config["model_sample_rate"] // 16000 for base_fft_size in wandb_config["16k_loss_fft_sizes"]]


    wandb_config = {**wandb_config,
    'batch_size': 1 if wandb_config['overfitting_test'] or wandb_config['dry_run'] else wandb_config['batch_size'],
    }

    assert not (wandb_config['use_cached_preprocessed_dataset'] and wandb_config['dry_run']), "dry run and use cached dataset are incompatible"
    assert not (wandb_config['use_cached_preprocessed_dataset'] and wandb_config['overfitting_test']), "overfitting test and use cached dataset are incompatible"

    print("WANDB CONFIG")
    print(json.dumps(wandb_config, indent=4))
    print("")


    # init wandb
    wandb.init(project="neural guitar, synthesis", config=wandb_config, notes=wandb_config["description"])

    torch.manual_seed(wandb.config.seed)
    np.random.seed(wandb.config.seed)
        
    trn_ds = data.load_prepared_data(prepared_data_path="./artefacts/guitarset_dataset_data_trn.pt", seconds_per_clip=wandb.config.n_seconds, sample_rate = wandb.config.model_sample_rate, feature_frame_rate=wandb.config.model_ft_frame_rate, use_random_offset=True)
    val_ds = data.load_prepared_data(prepared_data_path="./artefacts/guitarset_dataset_data_val.pt", seconds_per_clip=wandb.config.n_seconds, sample_rate = wandb.config.model_sample_rate, feature_frame_rate=wandb.config.model_ft_frame_rate, use_random_offset=False)
   
    example = trn_ds[0]
    n_samples = example["hex_audio"].shape[-1]

    if wandb.config.overfitting_test:
        DEMO_INTERVAL = 1024
        trn_ds = util.RepeatDataset(trn_ds, DEMO_INTERVAL * wandb.config.batch_size, 0)
    else:
        LIMIT_TRN_BATCHES = None

    wandb.config.n_samples = n_samples

    # convert wandb config to dict
    print(wandb.config.as_dict())


    model = SynthesisBase(
        wandb.config.as_dict(),
    )

    wandb_logger = WandbLogger(project="neural guitar, synthesis", log_model=True, notes=wandb.config.description)
    wandb_logger.watch(model, log="all")
    wandb.run.log_code(".")

    trn_dl = DataLoader(trn_ds, batch_size=wandb.config.batch_size, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=wandb.config.batch_size, shuffle=False, drop_last=False)

    checkpoint_callback = ModelCheckpoint(
        monitor='loss/val_epoch',
        dirpath='./artefacts/synthesis_checkpoints',
        filename=f"{wandb_logger.experiment.name}",
        save_top_k=1,
        mode='min',
    )

    trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=-1,
        accelerator='gpu' if wandb.config.gpu is not None else None,
        devices=[wandb.config.gpu],
        precision=16 if wandb.config.half_precision else 32,
        callbacks=[LearningRateMonitor(logging_interval='step'), checkpoint_callback],
        gradient_clip_val=1.0,
    )

    # pretty print wandb config
    print("wandb config:")
    for k,v in wandb.config.items():
        print(f"{k}: {v}")


    # reinit weights by recursively reinitializing all modules
  
    
    # model = SynthesisBase.load_from_checkpoint("./artefacts/synthesis_checkpoints/512 z, 64 frames, 64 harmonic, 64 noise bands, dilations up to 16, 6 stacks-v2.ckpt")
    # model.apply(weights_init)

    # print(model.hparams)
    # asd
    trainer.fit(model, trn_dl, val_dl)