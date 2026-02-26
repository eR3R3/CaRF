<p align="center">
  <h1 align="center">CaRF: Camera-Aware Referring Field for Multi-View Consistent 3D Gaussian Segmentation</h1>>
 <p align="center">
    <img src='https://img.shields.io/badge/Paper-Anonymized-grey?style=flat&logo=arXiv&' alt='Anonymous Paper'>
  </p>
</p>

---


This paper presents **Camera-Aware Referring Field (CaRF)**, a novel framework for referring segmentation in 3D Gaussian splatting that explicitly models view-dependent geometry and multi-view consistency. CaRF introduces two key components: (1) **In-Training Paired-View Supervision (ITPVS)**, which enforces consistency across calibrated camera views by jointly supervising Gaussian projections; and (2) **Gaussian Field Camera Encoding (GFCE)**, which integrates camera parameters into the Gaussian–language feature space to enhance geometric reasoning.  Extensive experiments on the **Ref-LERF**, **LERF-OVS**, and **3D-OVS** datasets demonstrate that CaRF achieves consistent improvements over existing methods while maintaining robust cross-view alignment and fine-grained semantic precision.  

![CaRF Framework Overview](framework.png)

---

## Datasets
To prepare the Ref-LERF dataset, please follow the directory structure below:
```bash
<path to ref-lerf dataset>
|---figurines
|---ramen
|---waldo_kitchen
|---teatime
...
```

## Checkpoints and Pseudo Masks

Pretrained checkpoints and pseudo masks will be released upon publication. 
All pseudo masks are generated following the [Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) pipeline.


## Setup

The recommended environment setup uses Conda:

```bash
conda env create --file environment.yml
conda activate carf
```

## Training

Before training CaRF, pretrain 3D Gaussian Splatting (3DGS) for RGB reconstruction as in [Gaussian Splatting (CVPR 2023)](https://github.com/graphdeco-inria/gaussian-splatting).

Then train CaRF with:
```bash
python train.py -s <path to dataset> -m <path to output_model>

<ref-lerf>
|---<path to dataset>
|   |---figurines
|   |---ramen
|   |---...
|---<path to output_model>
    |---figurines
    |---ramen
    |---...
```


## Rendering

To render referring results or open-vocabulary masks:

```bash
python render.py -m <path to output_model>
```


## Acknowledgments

We would like to thank the authors of ReferSplat for releasing their code and dataset publicly, which enabled us to conduct fair comparisons and extend their framework. The GitHub repository is available at:
https://github.com/heshuting555/ReferSplat.

We also acknowledge the open-source contributions of Grounded-SAM, BERT, and other foundational tools which our implementation builds upon.

Finally, we are grateful to our colleagues and reviewers for their constructive feedback which improved this work.
