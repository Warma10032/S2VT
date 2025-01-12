
## Description

This repository is used to document code for myself using S2VT to complete a kaggle competition about video describing.

The comptition link is [SEU_深度学习与应用2024 | Kaggle](https://www.kaggle.com/competitions/seu-2024/). It is a project in the Deep Learning Course SEU.

### For SEUer:

My code may have many issues and not perform well. If your teacher provides a file `Jupyter_Notebook-Video Caption`, that file is a better implementation of ideas and learning materials.

### For researcher who want to use S2VT in pytorch:

Most of my code refers to [YiyongHuang/S2VT](https://github.com/YiyongHuang/S2VT). And I deleted a lot of code that I didn't use in my little experiments. If you want to get more complete code, you can check the [reference repository](https://github.com/YiyongHuang/S2VT).

The most significant change to my code in `extract_features.py`. I used pytorch to load the VGG model instead of coffe. It's extremely convenient.


## Requirements 

Here is my python environment

* python = 3.11
* pytorch = 2.2.2 + cu118
* numpy
* opencv
* imageio
* scikit-image

## Running instructions

1. Install all the packages mentioned in the 'Requirements' section for the smooth running of this project.
2. Download the dataset from kaggle by using  `kaggle competitions download -c seu-2024` and put it into dataset/
3. Change all the path in these python files to point to directories in your workspace
4. Run extract_features.py to extract the RGB features of videos
5. Run train.py to train the model
6. Run test.py to generate the caption of test videos

## Dataset

The dataset is a chinese news video caption dataset with video, audio and sign language. It isn't an open dataset, if the kaggle link is failure, I have no idea.


## Acknowledgement

* S2VT: [chenxinpeng/S2VT](https://github.com/chenxinpeng/S2VT)
* Reference repository: [YiyongHuang/S2VT](https://github.com/YiyongHuang/S2VT)
* Dataset builder: [SEU dissertation](http://223.3.67.16/tpi651/Detail?dbID=6&dbName=SSLW&sysID=187060)
