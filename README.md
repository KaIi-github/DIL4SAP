## Progressive Adapting and Pruning: Domain-Incremental Learning for Saliency Prediction

This repository contains the original **PyTorch** implementation of **"Progressive Adapting and Pruning: Domain-Incremental Learning for Saliency Prediction"**, a new approach aims to design saliency prediction models that can imitate the incremental learning ability of human beings on multiple image domains.

## Dataset

 [SALICON](http://salicon.net/challenge-2017/)

 [MIT1003](https://people.csail.mit.edu/tjudd/WherePeopleLook/index.html)

 [CAT2000](http://saliency.mit.edu/results_cat2000.html)

 [WEBSAL](https://www-users.cse.umn.edu/~qzhao/webpage_saliency.html)

## Set Up Environment

To run this code set up python environment as follows:

```
https://github.com/KaIi-github/DIL4SAP.git
cd DIL4SAP
conda create -n DIL4SAP python=3.8
source activate DIL4SAP 
pip install -r requirements.txt
```

We tested our code with `Python 3.8` and `Cuda 11`.

## Repository Structure

Below are the main directories in the repository:

- `checkpoints/`: training checkpoints.
- `datasets/`: dataset repositories.
- `logs/`: records of results.
- `models/`: model definition and primary code.
- `model_hub/`: pretrained models.
- `scripts/`: scripts for train , inference and calculation of metrics.

Below are the main executable scripts in the repository:

- `train_new_task_step1.py`: script of pre-training for the first step.
- `train_salicon_1.py`: main training, pruning, and fine-tuningscript on salicon dataset.
- `train_art_2.py`: main training, pruning, and fine-tuning script on art dataset.
- `train_websal_3.py`: main training, pruning, and fine-tuning script on websal dataset.
- `predict_rap.py`: main inference script.
- `eval_command_rap.py`:main metric calculation script.
- `run_ours_sequence.sh`: bash script to run training
- `run_predict_rap.sh`: bash script to run inference and metric calculation.

## Running the Code

### Training

#### Data Format
The dataset tree folder references

```
├── Salicon
│   ├── trainSet
│   │   ├── Annotations
│   │   └── images
│   ├── testSet
│   │   ├── Annotations
│   │   └── images
|   |
├── MIT1003
│   ├── trainSet
│   │   ├── FIXATIONMAPS
│   │   └── STIMULI
│   ├── testSet
│   │   ├── FIXATIONMAPS
│   │   └── STIMULI
|   |
├── CAT2000
│   ├── trainSet
│   │   ├── FIXATIONMAPS
│   │   │   └── Art
│   │   └── STIMULI
│   │       └── Art
│   └── testSet
│       ├── FIXATIONMAPS
│       │   └── Art
│       └── STIMULI
│           └── Art
└── WebSal
    ├── trainSet
    │   ├── Annotations
    │   └── images
    └── testSet
        ├── Annotations
        └── images
```
#### Run Training
```
/bin/bash {Your Project Path}/run_ours_sequence.sh
```
### Inference
```
/bin/bash {Your Project Path}/run_predict_rap.sh
```
#### Model weights

To use pre-trained **ERFNet** download weights [here](https://github.com/Eromera/erfnet_pytorch/blob/master/trained_models/erfnet_encoder_pretrained.pth.tar).

## Citation

This work is currently accepted by Transactions on Multimedia Computing, Communications, and Applications (TOMM).

```
@article{10.1145/3661312,
  author = {Yang, Kaihui and Han, Junwei and Guo, Guangyu and Fang, Chaowei and Fan, Yingzi and Cheng, Lechao and Zhang, Dingwen},
  title = {Progressive Adapting and Pruning: Domain-Incremental Learning for Saliency Prediction},
  journal = {ACM Trans. Multimedia Comput. Commun. Appl.},
  year = {2024}
}
```

## References

## Acknowledgements
Thanks to Yingzi for her contribution to this work.

Thanks to [PackNet](https://github.com/arunmallya/packnet) and [MDIL](https://github.com/prachigarg23/MDIL-SS) for their work.

