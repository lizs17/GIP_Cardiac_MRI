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

# A simple reconstruction example for Poisson sampling
Just run the following command to train a GIP model and perform reconstruction from the very beginning (from randomly-initialized model weight).
```bash
python GIP_Poisson____main.py
```
[Reproduction on a 4090 GPU] The pretraining takes 2~3 hours and the ADMM algorithm takes ~1 hour.
[After ]
