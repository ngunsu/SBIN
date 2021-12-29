# SBIN: A stereo disparity estimation network using binary convolutions
This is the implementation of our paper"[SBIN: A stereo disparity estimation network using binary convolutions](.)"

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

#### Docker (training on x64 arch)

```bash
./docker/launch.sh

# Sceneflow training
python experiments.py train 2

# Kitti2012 (using sceneflow pretrained)
python experiments.py train 1
```

#### Further experiments

To add new experiments, check the experiments folder of directly use the cli.py, after reading the command line help

## Citation
```
```

