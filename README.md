# Bayesian Inverse Graphics (BIG)
This repository contains the code for the paper "Bayesian Inverse Graphics for Few-Shot Concept Learning"

TLDR: `probabilistic programming` + `differentiable rendering` = `minimal-data learning`

## Modules 
All modules are implemented in ```jax```

* [jaynes](https://github.com/oarriaga/bayesian-inverse-graphics/tree/main/jaynes) Probabilistic Programming Library (Automatic Bayesian Inference).
* [tamayo](https://github.com/oarriaga/bayesian-inverse-graphics/tree/main/tamayo) Differentiable Rendering Library.
* [lecun](https://github.com/oarriaga/bayesian-inverse-graphics/tree/main/lecun) Convnets.

## Run

### Setup
0. Install requirements e.g. `pip install -r requirements.txt` 
1. Download the datasets (fscvlr.zip) and weights (VGG16.eqx) from [here](https://github.com/oarriaga/bayesian-inverse-graphics/releases/tag/untagged-8ca632b1739c92a53c41).
2. Move `fsclvr.zip` inside repository `bayesian-inverse-graphics/`.  
3. Move `VGG16.eqx`  inside repository `bayesian-inverse-graphics/`.  
4. Extract datasets `unzip fsclvr.zip`

### Training
5. Run `python optimize_scene.py`
5. Run `python extract_features.py`
7. Run `python optimize_bijectors.py`

### Test
8. Run `python learn_concept.py --concept 0`

