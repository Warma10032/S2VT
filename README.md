## Requirements 

Here is my python environment

* python = 3.11
* pytorch = 2.2.2 + cu11.8
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
