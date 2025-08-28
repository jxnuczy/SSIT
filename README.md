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
