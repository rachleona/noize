
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
$ noize [OPTIONS] [FILEPATH] [OUTPUT_DIR]
```

**Arguments**:

* `[FILEPATH]`
* `[OUTPUT_DIR]`

**Options**:

* `--config-file PATH`: absolute path to config file to use [default: &LT;PATH_TO_PACKAGE&GT;/config.json]
* `--output-filename TEXT`: custom name for the protected output file [default: protected_&LT;FILEPATH&GT;]
* `--cdpam TEXT`: whether to use the cdpam quality term or not [default: True]
* `--avc / --no-avc`: whether to use the adaIN speaker encoder as part of perturbation generation [default: avc]
* `--freevc / --no-freevc`: whether to use the freeVC speaker encoder as part of the perturbation generation [default: freevc]
* `--yourtts / --no-yourtts`: whether to use the YourTTS speaker encoder as part of the perturbation generation [default: yourtts]
* `--perturbation-level INTEGER`: hard limit on magnitude of protective noise, recommended level between 5 to 30 [default: 5]
* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.

  
<div align="right">

  

[&nwarr; Back to top](#readme-top)

  

</div>

 **Additional Options for Development and Research Use**:
* `--cdpam-weight INTEGER`: [default: 50]
* `--distance-weight INTEGER`: [default: 2]
* `--snr-weight FLOAT`: [default: 0.005]
* `--perturbation-norm-weight FLOAT`: [default: 0.05]
* `--frequency-weight FLOAT`: [default: 0.3]
* `--avc-weight FLOAT`: [default: 1]
* `--freevc-weight FLOAT`: [default: 1]
* `--yourtts-weight FLOAT`: [default: 1]
* `--learning-rate FLOAT`: [default: 0.02]
* `--iterations INTEGER`: [default: 500]
* `--logs / --no-logs`: [default: no-logs]
* `--log-file TEXT`: [default: log.csv]
* `--target TEXT` [default: None]


### Python
Example usage in python:
```py
from rachleona_noize.main import main as noize
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
- Chou et al. for their work on [adaIN][avc]
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


