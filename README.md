# DDSP Guitar

This repository contains demo code for [*DDSP-based Neural Waveform Synthesis of Polyphonic Guitar Performance from String-wise MIDI Input*](https://arxiv.org/abs/2309.07658). The project was conducted together Xin Wang, Erica Cooper, Lauri Juvela, Bob L. T. Sturm and Junichi Yamagishi.

Audio examples here: https://erl-j.github.io/neural-guitar-web-supplement/.

## Setup

```
git clone https://github.com/erl-j/ddsp-guitar.git
cd ddsp-guitar
pip install -r requirements.txt
```

## Usage

For rendering a midi file with the *unified* model, run:

```bash
python render_midi.py --midi_path midi.mid --output_path out.wav --crop-seconds 10 --device cuda:0
```

Use ```--help``` for more options.

## Acknowledgements

This work is supported by MUSAiC: Music at the Frontiers of Artificial Creativity and Criticism (ERC-2019-COG No. 864189) and
JST CREST Grants (JPMJCR18A6 and JPMJCR20D3) and MEXT
KAKENHI Grants (21K17775, 21H04906, 21K11951, 22K21319).

DDSP code is based on [magenta's ddsp](https://github.com/magenta/ddsp) and [ddsp_pytorch](https://github.com/acids-ircam/ddsp_pytorch).
Code for dilated convolutions is from @ljuvela.



