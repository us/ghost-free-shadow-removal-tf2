# Towards Ghost-free Shadow Removal 

This repo contains the code and results of the AAAI 2020 paper:

<i><b>Towards Ghost-free Shadow Removal via <br> Dual Hierarchical Aggregation Network and Shadow Matting GAN</b></i><br>
[Xiaodong Cun](vinthony.github.io), [Chi-Man Pun<sup>*</sup>](http://www.cis.umac.mo/~cmpun/) ,Cheng Shi <br>
[University of Macau](http://um.edu.mo/)

[Syn. Datasets](#Resources) | [Models](#Resources) | [Results](#Resources) | [Paper]() | [Supp.]()

![remove_detail](https://user-images.githubusercontent.com/4397546/69003615-582b2180-0940-11ea-9faa-2f2ae6b1d5ba.png)

<i>We plot a result of our model with the input shown in yellow square. From two zoomed regions, our method removes the shadow and reduces the ghost successfully.</i>

## **Introduction**
<p style="text-align:justify"><i>Shadow removal is an essential task for scene understanding. Many studies consider only matching the image contents, which often causes two types of ghosts: color in-consistencies in shadow regions or artifacts on shadow boundaries. In this paper, we try to tackle these issues in two aspects. On the one hand, to carefully learn the border artifacts-free image, we propose a novel network structure named the Dual Hierarchically Aggregation Network(DHAN). It contains a series of growth dilated convolutions as the backbone without any down-samplings, and we hierarchically aggregate multi-context features for attention and prediction respectively. On the other hand, we argue that training on a limited dataset restricts the textural understanding of the network, which leads to the shadow region color in-consistencies. Currently, the largest dataset contains 2k+ shadow/shadow-free images in pairs. However, it has only 0.1k+ unique scenes since many samples share exactly the same background with different shadow positions. Thus, we design a Shadow Matting Generative Adversarial Network~(SMGAN) to synthesize realistic shadow mattings from a given shadow mask and shadow-free image. With the help of novel masks or scenes, we enhance the current datasets using synthesized shadow images. Experiments show that our DHAN can erase the shadows and produce high-quality ghost-free images. After training on the synthesized and real datasets, our network outperforms other state-of-the-art methods by a large margin. </i></p>


## **Resources**

- Pre-trained Models: <b>[SRD](https://uofmacau-my.sharepoint.com/:u:/g/personal/yb87432_umac_mo/EVjCDVbv4AhAsco1IqCTCnoBMVJt-f6pIFU603G0EEZ5CA?e=DDvg2v) | [SRD+](https://uofmacau-my.sharepoint.com/:u:/g/personal/yb87432_umac_mo/EXoaeGxsGMRMsnCs25_4Z4wB2XKlSY7q-LlF5d3kFvU2eg?e=a3VrLy) | 
[ISTD](https://uofmacau-my.sharepoint.com/:u:/g/personal/yb87432_umac_mo/EdGF_2nSCZdMgbL0cz4aCt4BvEtAZ0xNsy81rloxJy5m7g?e=orI9i1) | 
ISTD+ </b>

- Results on Shadow Removal: <b>
[SRD](https://uofmacau-my.sharepoint.com/:u:/g/personal/yb87432_umac_mo/EeBoBAhnCMpClEW5Wb-MY88BgzTQYf7-hDCnNrfmX_zevg?e=xu8AEh) | 
[SRD+](https://uofmacau-my.sharepoint.com/:u:/g/personal/yb87432_umac_mo/EYLodBImcw1AlfQZsh71HuYB_TalzP0uTBEtS-9atEdc_Q?e=DODEKk) | 
[ISTD](https://uofmacau-my.sharepoint.com/:u:/g/personal/yb87432_umac_mo/EQgDUC1d_BpFg7SCRDCAlTkBRDeKeATnbwYvVMCdkpWRBw?e=kxyrAE) | 
[ISTD+](https://uofmacau-my.sharepoint.com/:u:/g/personal/yb87432_umac_mo/ERvQFx8d8AxLmrafMi609nMBo7JnsV4a4s63FV_NP89_eA?e=13NIpd) 
</b>

- Results on Shadow Detection: <b> 
[SBU](https://uofmacau-my.sharepoint.com/:u:/g/personal/yb87432_umac_mo/EYByu0IMTQFHl__lK7GA1DAB0crwq0i49SIVLcdQWmnq_w?e=XO5OHg) | 
[SBU+](https://uofmacau-my.sharepoint.com/:u:/g/personal/yb87432_umac_mo/EQMCyGNUo3xJg8fInF7LWQAB0g9HFZHRBuBoxlzEL5CNUg?e=ENfsZV) </b>

- Training on ISTD dataset and generating shadow using USR dataset: <b> 
[Syn. Shadow](https://uofmacau-my.sharepoint.com/:u:/g/personal/yb87432_umac_mo/EW8-rjV5MX5BtoNSoDuzQg8B2lk4QHZS9jZzDDPfrEZVfg?e=DxPVfR) </b>

## **Setup**
Creating the conda environments following [here](https://github.com/ceciliavision/perceptual-reflection-removal#conda-environment).

## **Demo**

You can start a [jupyter](https://jupyter.org/) server and run the demo code following the instructions in `demo.ipynb`

It has been tested both in MacOS 10.15 and Ubuntu 18.04 LTS. Both CPU and GPU are supported (But Running on CPU is quite slow).



## **Training (TBD)**

### 1. Training to generate Synthesized Shadow

### 2. Training on the ISTD/SRD dataset

### 3. Training with data augmentation


## **Acknowledgements**
The author would like to thanks Nan Chen for her helpful discussion.

Part of the code is based upon [FastImageProcessing](https://github.com/CQFIO/FastImageProcessing) and [Perception Reflection Removal](https://github.com/ceciliavision/perceptual-reflection-removal)

## **Citation**

If you find our work useful in your research, please consider citing:

```
@inproceedings{cun2020shadow,
    author = {Xiaodong, Cun and Chi-Man, Pun and Cheng Shi},
    title = {Towards Ghost-free Shadow Removal via Dual Hierarchical Aggregation Network and Shadow Matting GAN},
    booktitle = {Proceedings of the AAAI},
    year = {2020}
}
```

## **Contact**
Please contact me if there is any question (Xiaodong Cun yb87432@um.edu.mo)