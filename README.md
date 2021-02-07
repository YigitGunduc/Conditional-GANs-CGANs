# Conditional Generative Adversarial Nets/CGANs

![cgan](https://cdn-images-1.medium.com/max/698/0*L8loWBQIJoUrPR00.png)
## Overview
This project implements Conditional GANs in Tensorflow & Keras to generate images according to a given label 

Based on the following paper
* https://arxiv.org/abs/1411.1784

## Usage and Getting Started
```
# clone the repo
git clone https://github.com/YigitGunduc/Spectrum.git

# install requirements
pip install -r requirements.txt

# training the model
python3 train.py --epochs EPOCHS --dataset mnist & fashion_mnist

# generating images
python3 generate.py LABEL
```
## Visualizations

![cgan](https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/3682850e-dc4d-4c07-a2c8-4e58a721b65b/f50369fd-32ea-477d-b74c-1b3f6e014122/images/screenshot.gif)

Source for GIF(https://www.mathworks.com/matlabcentral/fileexchange/75441-conditional-gan-and-cnn-classification-with-fashion-mnist)
