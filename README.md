[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/10QrhxfWa_F3VuywowFeogfxbpok7-ZX7?usp=sharing)


# Conditional Generative Adversarial Nets/CGANs

![cgan](https://cdn-images-1.medium.com/max/698/0*L8loWBQIJoUrPR00.png)
## Overview
This project implements Conditional GANs in Tensorflow & Keras to generate images according to a given label 

Based on the following paper
* https://arxiv.org/abs/1411.1784

## Usage and Getting Started
```
# clone the repo
git clone https://github.com/YigitGunduc/Conditional-GANs-CGANs.git

# install requirements
pip install -r requirements.txt

# training the model
python3 train.py

# generating images
python3 generate.py LABEL
```
## Visualizations

![cgan](https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/3682850e-dc4d-4c07-a2c8-4e58a721b65b/f50369fd-32ea-477d-b74c-1b3f6e014122/images/screenshot.gif)

Source for GIF(https://www.mathworks.com/matlabcentral/fileexchange/75441-conditional-gan-and-cnn-classification-with-fashion-mnist)



[contributors-shield]: https://img.shields.io/github/contributors/YigitGunduc/Conditional-GANs-CGANs.svg?style=flat-rounded
[contributors-url]: https://github.com/YigitGunduc/Conditional-GANs-CGANs/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/YigitGunduc/Spectrum.svg?style=flat-rounded
[forks-url]: https://github.com/YigitGunduc/repo/network/members
[stars-shield]: https://img.shields.io/github/stars/YigitGunduc/Conditional-GANs-CGANs.svg?style=flat-rounded
[stars-url]: https://github.com/YigitGunduc/repo/stargazers
[issues-shield]: https://img.shields.io/github/issues/YigitGunduc/Conditional-GANs-CGANs.svg?style=flat-rounded
[issues-url]: https://github.com/YigitGunduc/Conditional-GANs-CGANs/issues
