<p align="center">

  <h1 align="center">SuperDec: 3D Scene Decomposition with Superquadric Primitives</h1>
  <p align="center">
    <a href="https://elisabettafedele.github.io/">Elisabetta Fedele</a><sup>1,2</sup>,
    <a href="https://boysun045.github.io/boysun-website/">Boyang Sun</a><sup>1</sup>,
    <a href="https://geometry.stanford.edu/?member=guibas">Leonidas Guibas</a><sup>2</sup>, 
    <a href="https://people.inf.ethz.ch/pomarc/">Marc Pollefeys</a><sup>1,3</sup>, 
    <a href="https://francisengelmann.github.io/">Francis Engelmann</a><sup>2</sup>
    <br>
    <sup>1</sup>ETH Zurich, 
    <sup>2</sup>Stanford University, 
    <sup>3</sup>Microsoft <br>
  </p>
  <h2 align="center">ICCV 2025</h2>
  <h3 align="center"><a href="https://github.com/elisabettafedele/superdec">Code</a> | <a href="https://arxiv.org/abs/2504.00992">Paper</a> | <a href="https://super-dec.github.io">Project Page</a> </h3>
  <div align="center"></div>
</p>
<p align="center">
  <a href="">
    <img src="https://super-dec.github.io/static/figures/compressed/teaser/room0_1_bg.jpeg" alt="Logo" width="100%">
  </a>
</p>
<p align="center">
<strong>SuperDec</strong> allows to represent arbitrary 3D scenes with a compact and modular set of superquadric primitives.
</p>
<br>

---
## Setup 
Clone the repository, create virtual environment and install the required packages as follows:
```bash
python3 -m venv .venv  # create new virtual environment
source .venv/bin/activate  # activate it
pip install -r requirements.txt  # install requirements
pip install -e .  # install current repository in editable mode
python setup_sampler.py build_ext --inplace # -> only required for training
```

---
## Download ShapeNet
You can download the ShapeNet dataset (73.4 GB) by running:
```bash
bash scripts/download_shapenet.sh
```
After, you should have the dataset in `data/ShapeNet` folder. 

---
## Train on ShapeNet
To start a training on the ShapeNet dataset you can just run 
```bash
python trainer/train.py
```

---
## Run on objects
You can use the pretrained checkpoints to run SuperDec on arbitrary objects. The only thing you need is a point cloud dataset!

## Acknowledgement
We adapted some codes from some awesome repositories including [superquadric_parsing](https://github.com/paschalidoud/superquadric_parsing), [CuboidAbstractionViaSeg](https://github.com/SilenKZYoung/CuboidAbstractionViaSeg), [volumentations](https://github.com/kumuji/volumentations), [LION](https://github.com/nv-tlabs/LION), [occupancy_networks](https://github.com/autonomousvision/occupancy_networks), and [convolutional_occupancy_networks](https://github.com/autonomousvision/convolutional_occupancy_networks). Thanks for making codes and data public available. 


## TODO

- [ ] Code from pth to npz