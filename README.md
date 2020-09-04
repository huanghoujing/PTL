# About

This is the official implementation of paper **Proxy Task Learning For Cross-Domain Person Re-Identification**, ICME 2020. 

```
@inproceedings{huang2020proxy,
  title={Proxy Task Learning For Cross-Domain Person Re-Identification},
  author={Huang, Houjing and Chen, Xiaotang and Huang, Kaiqi},
  booktitle={2020 IEEE International Conference on Multimedia and Expo (ICME)},
  pages={1--6},
  year={2020},
  organization={IEEE}
}
```

# Requirements

- Python 2.7
- Pytorch 1.0.0
- Torchvision 0.2.1
- No special requirement for sklearn version

# Dataset Path

Prepare datasets to have following structure:
```
${project_dir}/dataset
    market1501
        Market-1501-v15.09.15
        Market-1501-v15.09.15_ps_label
    cuhk03_np_detected_jpg
        cuhk03-np  # Extracted from cuhk03-np.zip, https://pan.baidu.com/s/1RNvebTccjmmj1ig-LVjw7A
        cuhk03-np-jpg_ps_label
    duke
        DukeMTMC-reID
        DukeMTMC-reID_ps_label
    peta
        PETA dataset
            3DPeS
            CAVIAR4REID
            ...
    rap
        RAP_dataset
        RAP_annotation
    pa100k       # Not used in this project
        PA-100K  # https://drive.google.com/drive/folders/0B5_Ra3JsEOyOUlhKM0VPZ1ZWR2M
            annotation.mat
            release_data
                release_data
```

- `Market-1501-v15.09.15_ps_label`, `cuhk03-np-jpg_ps_label` and `DukeMTMC-reID_ps_label` can be downloaded from [Baidu Cloud](https://pan.baidu.com/s/1Mm2gWO-Xg3wiyCd6SEAWaA) or [Google Drive](https://drive.google.com/open?id=1BARSoobjTAPeOSOM-HnGzlOYTj1l9-Qs).
- PETA can be downloaded from http://mmlab.ie.cuhk.edu.hk/projects/PETA.html. By `unzip PETA.zip`, you will obtain `PETA dataset`.
- RAP (v2) can be downloaded according to [license agreement](https://drive.google.com/open?id=1hoPIB5NJKf3YGMvLFZnIYG5JDcZTxHph). Please refer to [this repository](https://github.com/dangweili/pedestrian-attribute-recognition-pytorch).

# Testing

Trained Baseline and PTL models can be downloaded from [Baidu Cloud](https://pan.baidu.com/s/1emfrCyJzGET64lcbWWzvhQ) (password `1bvf`) or [Google Drive](https://drive.google.com/drive/folders/1e4WdVXmqO7WSrj6Pju-NwqYaWAM1JQMb?usp=sharing). Place the `exp` folder under the project directory.

For example,
- Test baseline, Market1501 to CUHK03 or Duke.
    ```bash
    gpu=0 only_test=True dataset=market1501 bash script/train_baseline.sh
    ```
    This should give result
    ```
    M -> D      [mAP:  29.1%], [cmc1:  49.9%], [cmc5:  63.8%], [cmc10:  69.9%]
    ```

- Test PTL, Market1501 to Duke.
    ```bash
    gpu=0 only_test=True src_dset=market1501 tgt_dset=duke bash script/train_ptl.sh
    ```
    This should give result
    ```
    M -> D      [mAP:  36.2%], [cmc1:  57.4%], [cmc5:  71.0%], [cmc10:  75.8%]
    ```

# Prepare Attribute

Train attribute recognition on PETA,
```bash
gpu=0 bash script/train_attr.sh
```
The script will also evaluate on validation set.

Predict soft attribute labels on ReID images,
```bash
gpu=0 predict_on_dataset=market1501 bash script/predict_attr.sh;
gpu=0 predict_on_dataset=cuhk03_np_detected_jpg bash script/predict_attr.sh;
gpu=0 predict_on_dataset=duke bash script/predict_attr.sh;
```
The results will be saved inside `${project_dir}/dataset/predicted_attr/...`.

The predicted attribute labels can also be downloaded from [Baidu Cloud](https://pan.baidu.com/s/1emfrCyJzGET64lcbWWzvhQ) (password `1bvf`) or [Google Drive](https://drive.google.com/drive/folders/1e4WdVXmqO7WSrj6Pju-NwqYaWAM1JQMb?usp=sharing). 

# Train Baseline

Train baseline on Market1501
```bash
gpu=0 dataset=market1501 bash script/train_baseline.sh
```

# Proxy Task Learning

For `Market1501 -> Duke`,
```bash
gpu=0 src_dset=market1501 tgt_dset=duke bash script/train_ptl.sh
```

# Ablation Study

Ablation can be found in `script/train_ptl.sh`.


# Visualize Human Parsing (Part Segmentation) Labels

```bash
python script/save_colorful_ps_label.py
```
