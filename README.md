# SSIT
SSIT: A Spatial-Spectral Interactive Transformer for Hyperspectral Image Denoising

<hr />

> **Abstract:** 
Convolutional neural networks have been successfully applied to hyperspectral image denoising, but they cannot effectively capture global information in
the image. To address this issue, spatial-spectral transformers have recently
been proposed to capture global spectral correlation and non-local spatial
similarity. Nevertheless, these solutions cannot capture essential spatialspectral characteristics in hyperspectral images well and fuse spatial and
spectral features effectively. Based on this, we propose a spatial-spectral interactive transformer for hyperspectral image denoising. This method comprises a spatial-spectral attention module and a spatial-spectral interactive
module. The former comprises a spatial permuted self-attention to model local and non-local spatial similarities and a spectral compression attention to
extract spectral features with global spectral correlation and low-rank property. The latter further integrates the spatial-spectral features interactively.
Additionally, we design a spectral-split feed-forward network to improve the
model’s capability to model spatial-spectral characteristics. Extensive experiments on synthetic and real noisy hyperspectral images demonstrate the
effectiveness of the proposed method.

<hr />

## Network Architecture

<img width="8828" height="6460" alt="SSIT_Model" src="https://github.com/user-attachments/assets/68070f95-f3e1-4e5c-80e2-56f12cb86db5" />



## Contents
1. [Datasets](#Datasets)
1. [Training and Testing](#Training)
1. [Results](#Results)


## Datasets

### ICVL Dataset
* The entire ICVL dataset download link: https://icvl.cs.bgu.ac.il/hyperspectral/

### Realistic Dataset
* Please refer to [[github-link]](https://github.com/ColinTaoZhang/HSIDwRD) for "Hyperspectral Image Denoising with Realistic Data in ICCV, 2021" to download the dataset

### Urban dataset
* The training dataset are from link: https://apex-esa.org/. The origin Urban dataset are from link:  https://rslab.ut.ac.ir/data.

### Houston2018 dataset
* The Houston2018 dataset are from link: .

## Training and Testing
### ICVL Dataset
```
#for gaussian noise
#----training----
python hside_simu.py -a ssit_base -p ssit_base_gaussian

#----testing---- 
python hside_simu_test.py -a ssit_base -p ssit_base_gaussian_test -r -rp checkpoints/icvl_gaussian.pth --test-dir /icvl_noise/512_50
```

## Acknowledgement
The codes are based on [SERT](https://github.com/MyuLi/SERT).

