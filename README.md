# DDSP Guitar

This repository contains demo code for [*DDSP-based Neural Waveform Synthesis of Polyphonic Guitar Performance from String-wise MIDI Input*](https://arxiv.org/abs/2309.07658)

## Setup

```
git clone https://github.com/erl-j/ddsp-guitar.git
cd ddsp-guitar
pip install -r requirements.txt
```

## Usage

For rendering a midi file, run:

```bash
python render_midi.py --midi_path midi.mid --output_path out.wav --crop-seconds 10 --device cuda:0
```

Use ```--help``` for more options.

## Acknowledgements

DDSP code is based on [magenta's ddsp](https://github.com/magenta/ddsp) and [ddsp_pytorch](https://github.com/acids-ircam/ddsp_pytorch).
Code for dilated convolutions is from @ljuvela.



