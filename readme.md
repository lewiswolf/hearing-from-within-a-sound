# Hearing from Within a Sound

![Max MSP version](https://img.shields.io/badge/Max-8+-blue)
![python version](https://img.shields.io/badge/Python-3.11-blue)

Max patch and python codebase used for exploring and demonstrating the work described in the AES paper _Hearing from Within a Sound: A Series of Techniques for Deconstructing and Spatialising Timbre_.

## Max Project

### Dependencies

-   [IRCAM Spat~ v.5+](https://forum.ircam.fr/projects/detail/spat)
-	[kac_maxmsp](https://github.com/lewiswolf/kac_maxmsp)

## Python Resynthesis

Demo video 🎥

[![Watch the video](https://i.ytimg.com/vi/-0i4IlHmgRs/maxresdefault.jpg)](https://youtu.be/-0i4IlHmgRs)

```bash
cd python
```

### Dependencies

-   [libsndfile](https://github.com/libsndfile/libsndfile)
-   [pipenv](https://formulae.brew.sh/formula/pipenv#default)
- 	[PyTorch with CUDA enabled](https://pytorch.org/get-started/locally/)

If you're using conda:

```bash
conda env create -f environment.yml
conda activate aes
```

### Install

```bash
pipenv install
```

### Run

You will need to specify a directory of audio files to resynthesise.

```bash
pipenv run python resynthesise.py --audio_dir </absolute/path/to/audio/files/>
```

### Testing

```bash
pipenv install -d
pipenv run test
```

## How to Cite

```bibtex
@inproceedings{wolstanholmeHearingSoundSeries2023,
  title = 	{Hearing from within a Sound: {{A}} Series of Techniques for Deconstructing and Spatialising Timbre},
  booktitle =	{International {{Conference}} on {{Spatial}} and {{Immersive Audio}} ({{AES}})},
  author = 	{Wolstanholme, Lewis and Vahidi, Cyrus and McPherson, Andrew},
  year = 	{2023},
  month = 	aug,
  address = 	{{Huddersfield, UK}},
  copyright = 	{All rights reserved}
}
```