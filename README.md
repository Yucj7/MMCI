# MMCI
This is the code of paper "Reinforcement Learning-Based Edge-Assisted Inference with Multimodal Data"

## file structure
MMCI/
├── Algorithm/                # reinforcement learning algorithms \n
├── env/                      # simulation environment \n
├── imagebind/                # imagebind model code \n
├── results/                  # the data results of reinforcement learning algorithms \n
├── shap/                     # calculate shap value \n
├── build_model.py            # build imagebind model \n
├── DMMCI_master.py           # DMMCI implement code \n
├── MMCI_master.py            # MMCI implement code \n
├── ReADME.md                 # illustrate documentation \n
├── requirements.txt          # python environment \n
├── runningTime.py            # running time of the model on FP16 and FP32 \n
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




