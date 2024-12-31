
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
![pytorch_badge]
[![black_badge]][black]
<!-- badges -->

<!--**&searr;&nbsp;&nbsp;Read my dissertation here&nbsp;&nbsp;&swarr;**

<a href="https://gowebly.org" target="_blank" title="Go to the Gowebly CLI website"><img width="99%" alt="gowebly create command" src="https://raw.githubusercontent.com/gowebly/.github/main/images/gowebly_create.gif"></a> -->

 
## ⚡️ Quick start

> [!IMPORTANT]
> Noize works best with GPUs as it requires a lot of computational power. Please ensure you have the necessary hardware before proceeding
  
Install using the following command
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


```console
$ noize [OPTIONS] [FILEPATH] [OUTPUT_DIR]
```

**Arguments**:

* `[FILEPATH] PATH` 
* `[OUTPUT_DIR] PATH`

**Options**:

* `--config-file PATH`: [default: config.json]
* `--output-filename TEXT`: [default: protected_]
* `--perturbation-level INTEGER`: [default: 5]
* `--cdpam-weight INTEGER`: [default: 50]
* `--distance-weight INTEGER`: [default: 2]
* `--learning-rate FLOAT`: [default: 0.02]
* `--iterations INTEGER`: [default: 500]
* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.
  

<div align="right">

  

[&nwarr; Back to top](#readme-top)

  

</div>

  

<!-- ## 📈 Performance


## 🔧 Configurations


## 🫡 Acknowledgement

-openvoice
-antifake -->

<!-- badge links -->

[part_ii_badge]:https://img.shields.io/badge/cambridge%20part%20ii-85B09A?style=for-the-badge
[pytorch_badge]:https://img.shields.io/badge/pytorch-2.2+-ee4c2c.svg?style=for-the-badge
[black_badge]:https://img.shields.io/badge/code%20style-black-000000.svg?style=for-the-badge

<!--package and repo links-->
[black]:https://github.com/psf/black

<!--other links-->
[part_ii_page]:https://www.cst.cam.ac.uk/teaching/part-ii/projects
[glaze]:https://glaze.cs.uchicago.edu/
[antifake]:https://sites.google.com/view/yu2023antifake


