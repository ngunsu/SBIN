# SBIN: A stereo disparity estimation network using binary convolutions
This is the implementation of our paper"[SBIN: A stereo disparity estimation network using binary convolutions](https://latamt.ieeer9.org/index.php/transactions/article/view/5909)"

---

## Results

| Model | Dataset | EPE | Err > 3 | Exp ID|
|:-----:|:-------:|:---:|:-------:|:------:|
| normalbin_res_prelu_avgpool| Kitti2012| 2.0979 | 0.1370 | 1 |

---

## Reproduce best results (Training in using docker)

### Requirements

- [docker](https://docs.docker.com/engine/install/ubuntu/)
- [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
- Download Kitti2012, Kitti2015 and Sceneflow datasets
    - Datasets must be inside the folder **datasets** in the **root folder** of this repository
    - Datasets
        - kitti2012
        - kitti2015
        - sceneflow

**Log and checkpoints are created on the folder output**

#### Docker (training on x64 arch)

```bash
./docker/launch.sh

# Sceneflow training
python experiments.py train 2

# Kitti2012 training 
python experiments.py train 1
```

#### Further experiments

To add new experiments, check the experiments folder of directly use the cli.py, after reading the command line help

## Citation
```
@article{Aguilera_2022,
title={SBIN: A stereo disparity estimation network using binary convolutions},
volume={20},
url={https://latamt.ieeer9.org/index.php/transactions/article/view/5909},
number={4}, journal={IEEE Latin America Transactions},
author={Aguilera, Cristhian Alejandro},
year={2022},
month={Jan.}, 
pages={693–699} }
```

