
<a name="readme-top"></a>

# Noize CLI

> [!NOTE]  
> Development in progress!

An easy-to-use CLI tool that uses **adversarial perturbation** to protect audio clips from being used in unauthorised voice cloning. Created as part of my final year project at the University of Cambridge. 

<!--gif?-->

Inspired by [AntiFake][antifake] and [Glaze][glaze].

<!-- badges -->
<!--remember to add python version here-->
[![part_ii_badge]][part_ii_page]
![python_badge]
![pytorch_badge]
[![black_badge]][black]
<!-- badges -->

<!--**&searr;&nbsp;&nbsp;Read my dissertation here&nbsp;&nbsp;&swarr;**

<a href="https://gowebly.org" target="_blank" title="Go to the Gowebly CLI website"><img width="99%" alt="gowebly create command" src="https://raw.githubusercontent.com/gowebly/.github/main/images/gowebly_create.gif"></a> -->

 
## ⚡️ Quick start

> [!IMPORTANT]
> Noize works best with GPUs as it requires a lot of computational power. Please ensure you have the necessary hardware before proceeding!

First, make sure you have [Git Large File Storage][git_lfs] installed as this repository includes some large model weights stored via Git LFS. If you don't have it, please download the package [here][git_lfs] and install via
```console
git lfs install --skip-repo
```

Then, install Noize using the following command
```console
pip install git+https://github.com/rachleona/noize.git
```

And check using
```console
noize --help
```

That's it! You are now ready to start using Noize :)

## 📖 Reference

<!-- > [!IMPORTANT]
> While Noize works on all platforms, it currently works best with GPUs as it requires a lot of computational power. If you don't have access to the recommended hardware, please read through the performance section and make sure you understand the limitations of the software before you proceed. -->

### Command line

**Usage**:

```console
$ noize [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.

**Commands**:

* `protect`: Apply adversarial perturbation to produce...
* `web`: Starts Flask server for web interface
* `voices`: Manage saved target voices to be used in...


<div align="right">

  

[&nwarr; Back to top](#readme-top)

  

</div>

#### `noize protect`

Apply adversarial perturbation to produce protected audio file

**Usage**:

```console
$ noize protect [OPTIONS] [FILEPATH] [OUTPUT_DIR]
```

**Arguments**:

* `[FILEPATH]`: path to the audio file to apply protection on
* `[OUTPUT_DIR]`: name of output directory, will be created if not already exists

**Options**:

* `--perturbation-level INTEGER`: values between 1-10, controls how strong the noise applied is  [default: 5]
* `--target TEXT`: id of a saved voice for target-based optimisation
* `--config-file PATH`: path to the config file, use official repo one by default
* `--output-filename TEXT`: name of protected file, uses protected_&lt;filpath&gt; by default
* `--avc / --no-avc`: whether to use adaptVC encoder in perturbation calculation  [default: avc]
* `--freevc / --no-freevc`: whether to use FreeVC encoder in perturbation calculation  [default: freevc]
* `--xtts / --no-xtts`: whether to use XTTS encoder in perturbation calculation  [default: xtts]
* `--logs / --no-logs`: whether to log intermediate loss values  [default: no-logs]
* `--log-file TEXT`: name of log file is logs are requested  [default: log.csv]
* `--resource-log PATH`: name of resource log file if logs are requested
* `--learning-rate FLOAT`: learning rate to use for optimisation process  [default: 0.02]
* `--iterations INTEGER`: optimisation iterations (per audio segment)  [default: 300]
* `--distance-weight FLOAT`: weight used in loss functions  [default: 2]
* `--snr-weight FLOAT`: weight used in loss functions  [default: 0.025]
* `--perturbation-norm-weight FLOAT`: weight used in loss functions  [default: 0.25]
* `--frequency-weight FLOAT`: weight used in loss functions  [default: 1.5]
* `--avc-weight FLOAT`: weight used in loss functions  [default: 25]
* `--freevc-weight FLOAT`: weight used in loss functions  [default: 25]
* `--xtts-weight FLOAT`: weight used in loss functions  [default: 25]
* `--help`: Show this message and exit.

#### `noize web`

Starts Flask server for web interface

**Usage**:

```console
$ noize web [OPTIONS]
```

**Options**:

* `--port INTEGER`: port for server to listen  [default: 5000]
* `--debug / --no-debug`: whether to start server in debug mode  [default: no-debug]
* `--help`: Show this message and exit.

#### `noize voices`

Manage saved target voices to be used in protection application

**Usage**:

```console
$ noize voices [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `list`: Lists the ids of all saved voices that can...
* `play`: Plays 10s of the reference clip given for...
* `add`: Add a new saved voice for use as...
* `reset`: Recalculates embeddings for a saved voice
* `resetall`: Recalculates embeddings for all saved voices
* `delete`: Deletes a saved voice from voice library
  

### Python
Example usage in python:
```py
from rachleona_noize.cli.main import protect as noize
noize("/path/to/audio.wav", "outputs/dir", output_filename="protected/audio.wav", perturbation_level=10)
```
All options available for the CLI are also accessible on Python!




  

<!-- ## 📈 Performance


## 🔧 Configurations-->


## 🫡 Acknowledgement
This project was created with several key project as reference and includes many components pulled from existing TTS/voice cloning projects. Here I would like to credit those behind the works and contributed heavily to this project:

- Shan et al. for creating [Glaze][glaze] and inspiring this project
- Yu et al. who were behind the [AntiFake project][antifake] which Noize is heavily based on
- Qin et al. who created the [OpenVoice framework][openvoice], which is a core part of Noize
- Chou et al. for their work on [adaptVC][avc]
- and contributors to the [Coqui TTS library][coqui_tts], which was a great resource during the development of this project



<div align="right">

  

[&nwarr; Back to top](#readme-top)

  

</div>

<!-- badge links -->

[part_ii_badge]:https://img.shields.io/badge/cambridge%20part%20ii-85B09A?style=for-the-badge
[pytorch_badge]:https://img.shields.io/badge/pytorch-2.2+-ee4c2c.svg?style=for-the-badge
[black_badge]:https://img.shields.io/badge/code%20style-black-000000.svg?style=for-the-badge
[python_badge]:https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Frachleona%2Fnoize%2Frefs%2Fheads%2Fmain%2Fpyproject.toml&style=for-the-badge

<!--package and repo links-->
[black]:https://github.com/psf/black
[git_lfs]:https://git-lfs.com/
[coqui_tts]:https://github.com/coqui-ai/TTS
[avc]:https://github.com/jjery2243542/adaptive_voice_conversion
[openvoice]:https://github.com/myshell-ai/OpenVoice

<!--other links-->
[part_ii_page]:https://www.cst.cam.ac.uk/teaching/part-ii/projects
[glaze]:https://glaze.cs.uchicago.edu/
[antifake]:https://sites.google.com/view/yu2023antifake


