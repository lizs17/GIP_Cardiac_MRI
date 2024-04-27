# GIP_Cardiac_MRI
A simple implementation of the GIP method: "Graph Image Prior for Unsupervised Dynamic MRI Reconstruction".

The Network Architecture:
![Image text](illustration/Network_Architecture.png)

The Optimization Algorithm:
![Image text](illustration/Optimization_Algorithm.png)

# Environment Configuration
The following environmental configurations can reproduce the results in the article. Other configurations may also be feasible but have not been attempted.

* python=3.8.19
* numpy=1.24.3
* pytorch=1.12.1
* scipy=1.10.1
* matplotlib=3.7.5
* scikit-image=0.21.0
* torchkbnufft=1.4.0
  
The supporting file "environment.txt" is provided for a quick configuration for the necessary packages.

# Contents
* "data/": the dynamic cine MRI data directory. Each data contains a fully-sampled image 'img' and the corresponding estimated sensitivity maps 'smap'
* "mask/": the undersampling pattern directory, containing the undersampling mask (for Cartesian-sampled k-space patterns) or trajectory and the corresponding density compensation funtion (for non-Cartesian-sampled k-space patterns)
* utis.py: the helper functions and classes
* model.py: the Graph-Image_Prior (GIP) generator architecture
* GIP_xxx____pretrain.py: the code for pretraining the generator G_{\theta}, using xxx trajectory sampling
* GIP_xxx____ADMM.py: the code of the ADMM algorithm for alternately optimizing the generator G_{\theta} and the dynamic images, using xxx trajectory sampling
* GIP_xxx____main.py: a one-click script for running the code to reproduce the results, using xxx trajectory sampling

# Updated Information
* Pretraining is critical for producing good reconstruction performance,

# A Simple Reconstruction Example for Poisson Sampling
Just run the following command to train a GIP model and perform reconstruction from the very beginning (from randomly-initialized model weight).

```bash
python GIP_Poisson____main.py
```

The pretraining takes 2~3 hours and the ADMM algorithm takes ~1 hour (accordig to the reproduction on a RTX 4090 GPU).

When the code-runing is finished, two additional directories ("GIP_Poisson_R16.0_ACS6x6/" and "output_Poisson/") should be created:

```bash
├── GIP_Poisson_R16.0_ACS6x6/
│   ├── fs_0032_3T_slc1_p3/
│   │   ├── ADMM/
│   │   ├── C/
│   │   ├── G/
│   │   ├── tune/
│   │   ├── z.h5
├── output_Poisson/
│   ├── imgGT.png
│   ├── imgRC1.png
│   ├── imgRC10.png
│   ├── imgRC20.png
│   ├── imgZF.png
│   ├── masks.png
│   ├── smaps.png
```
* The "GIP_Poisson_R16.0_ACS6x6/" directory contains the intermediate and final results of the GIP reconstruction procedure.
* The "output_Poisson/" directory contains the plotted figures of the reconstructed images.
* IMPORTANT!!! Because the "nn.Upsample" module has unavoidable randomness (even if all the random seeds have been controlled, and the torch backends has been kept as deterministic, see https://discuss.pytorch.org/t/non-deterministic-behavior-of-pytorch-upsample-interpolate/42842/6), the 
