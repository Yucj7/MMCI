# MMCI
This is the code of paper "Reinforcement Learning-Based Edge-Assisted Inference with Multimodal Data"

## file structure
MMCI/
├── Algorithm/                # reinforcement learning algorithms
├── env/                      # simulation environment
├── imagebind/                # imagebind model code
├── results/                  # the data results of reinforcement learning algorithms
├── shap/                     # calculate shap value
├── build_model.py            # build imagebind model
├── DMMCI_master.py           # DMMCI implement code
├── MMCI_master.py            # MMCI implement code
├── ReADME.md                 # illustrate documentation
├── requirements.txt          # python environment
├── runningTime.py            # running time of the model on FP16 and FP32
└── setup.py                  # imagebind dependencies                  


# Usage
## install
Please download ImageBind before using this code.For details, please refer to https://github.com/facebookresearch/ImageBind

## usage
```shell
conda create --name simdmmci python=3.10 -y
conda activate simdmmci

pip install -r requirements.txt
```
## run RL
MMCI
```shell
python MMCI_master.py
```
DMMCI
```shell
python DMMCI_master.py
```

## shap
calculate shap value
```shell
cd shap/
python deep_shap.py
```




