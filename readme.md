# Hearing from Within a Sound

![Max MSP version](https://img.shields.io/badge/Max-8+-blue)
![python version](https://img.shields.io/badge/Python-3.11-blue)
<!-- <a href="https://doi.org/10.5281/zenodo.7274474">
![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.7274474-blue)
</a> -->

Max patch and python codebase used for exploring and demonstrating the work described in the AES paper _Hearing from Within a Sound_.

## Max Project

### Dependencies

-   [IRCAM Spat~ v.5+](https://forum.ircam.fr/projects/detail/spat)
-	[kac_maxmsp](https://github.com/lewiswolf/kac_maxmsp)

## Python Resynthesis

```bash
cd python
```

### Dependencies

-   [libsndfile](https://github.com/libsndfile/libsndfile)
-   [pipenv](https://formulae.brew.sh/formula/pipenv#default)
- 	[PyTorch with CUDA enabled](https://pytorch.org/get-started/locally/)

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