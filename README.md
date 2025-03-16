# Bayesian Inverse Graphics (BIG)
This repository contains the code for the paper "Bayesian Inverse Graphics for Few-Shot Concept Learning"

TLDR: `probabilistic programming` + `differentiable rendering` = `minimal-data learning`

## Citation
Here are the links for the preprint version [https://arxiv.org/abs/2409.08351](https://arxiv.org/abs/2409.08351) and the [NeSy springer](https://link.springer.com/chapter/10.1007/978-3-031-71167-1_8) version.

```BibTeX
@inproceedings{arriaga2024bayesian,
  title={Bayesian Inverse Graphics for Few-Shot Concept Learning},
  author={Arriaga, Octavio and Guo, Jichen and Adam, Rebecca and Houben, Sebastian and Kirchner, Frank},
  booktitle={International Conference on Neural-Symbolic Learning and Reasoning},
  pages={141--165},
  year={2024},
  organization={Springer}
}
```

## Poster

<img src="https://raw.githubusercontent.com/oarriaga/bayesian-inverse-graphics/refs/heads/main/images/poster.png" width="1080">

## Modules 
All modules are implemented in ```jax```

* [jaynes](https://github.com/oarriaga/bayesian-inverse-graphics/tree/main/jaynes) Probabilistic Programming Library (Automatic Bayesian Inference).
* [tamayo](https://github.com/oarriaga/bayesian-inverse-graphics/tree/main/tamayo) Differentiable Rendering Library.
* [lecun](https://github.com/oarriaga/bayesian-inverse-graphics/tree/main/lecun) Convnets.

## Run

### Setup
0. Install requirements e.g. `pip install -r requirements.txt` 
1. Download the datasets (fscvlr.zip) and weights (VGG16.eqx) from [here](https://github.com/oarriaga/bayesian-inverse-graphics/releases/tag/v0.0.1).
2. Move `fsclvr.zip` inside repository `bayesian-inverse-graphics/`.  
3. Move `VGG16.eqx`  inside repository `bayesian-inverse-graphics/`.  
4. Extract datasets `unzip fsclvr.zip`

### Training
5. Run `python optimize_scene.py`
5. Run `python extract_features.py`
7. Run `python optimize_bijectors.py`

### Test
8. Run `python learn_concept.py --concept 0`



## Funding
This project was developed in the [Robotics Group](https://robotik.dfki-bremen.de/de/ueber-uns/universitaet-bremen-arbeitsgruppe-robotik.html) of the [University of Bremen](https://www.uni-bremen.de/), together with the [Robotics Innovation Center](https://robotik.dfki-bremen.de/en/startpage.html) of the **German Research Center for Artificial Intelligence** (DFKI) in **Bremen**.
It has been funded by the German Federal Ministry for Economic Affairs and Energy and the [German Aerospace Center](https://www.dlr.de/DE/Home/home_node.html) (DLR), in the [PhysWM](https://robotik.dfki-bremen.de/en/research/projects/physwm) project.
