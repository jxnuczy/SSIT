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

### ICVL 
* The entire ICVL dataset download link: https://icvl.cs.bgu.ac.il/hyperspectral/

### Realistic Dataset
* Please refer to [[github-link]](https://github.com/ColinTaoZhang/HSIDwRD) for "Hyperspectral Image Denoising with Realistic Data in ICCV, 2021" to download the dataset

### Urban dataset
* The training dataset are from link: https://apex-esa.org/. The origin Urban dataset are from link:  https://rslab.ut.ac.ir/data.

### Houston dataset

## Training and Testing
### ICVL Dataset
```
#for gaussian noise
#----training----
python hside_simu.py -a sert_base -p sert_base_gaussian

#----testing---- The results are shown in Table 1 in the main paper.
python hside_simu_test.py -a sert_base -p sert_base_gaussian_test -r -rp checkpoints/icvl_gaussian.pth --test-dir /icvl_noise/512_50
```

```
#for comlpex noise
#----training----
python hside_simu_complex.py -a sert_base -p sert_base_complex

#----testing---- The results are shown in Table 2 in the main paper.
python hside_simu_test.py -a sert_base -p sert_base_complex_test -r -rp checkpoints/icvl_complex.pth --test-dir  /icvl_noise/512_mix
```

### Urban Dataset
```
#----training----
python hside_urban.py -a sert_urban -p sert_urban 

#----testing----  The results are shown in Figure 4 in the main paper.
python hside_urban_test.py -a sert_urban -p sert_urban_test -r -rp ./checkpoints/real_urban.pth
```

### Realistic Dataset
```
#----training----
python hside_real.py -a sert_real -p sert_real

#----testing---- The results are shown in Table 3 in the main paper.
python hside_real_test.py -a sert_real -p sert_real_test -r -rp ./checkpoints/real_realistic.pth


## Acknowledgement
The codes are based on SERT, link: https://github.com/MyuLi/SERT.

