
# CDefficient

## Associated Publications
- (Under submission) Wang et al. (2025) Computational and Data-Efficient Deep Learning for Robust and Fast Object Detection and Classification

## Setup

#### Requirerements
- Ubuntu 18.04
- GPU Memory => 16 GB
- GPU driver version >= 530.30.02
- GPU CUDA >= 12.1
- Python (3.8.20), opencv-python (4.11.0.86), PyTorch (2.4.1), torchvision (0.19.1).

#### Download
Execution file, configuration file, and models are download from the [zip](https://drive.google.com/drive/folders/1qrTJaNxNXHD6w01SO676rCfqAvDCJR7P) file.

## Steps
#### 1.Installation

Please refer to the following instructions.
```
# create and activate the conda environment
conda create -n CDefficient_Detector python=3.8 -y
conda activate CDefficient_Detector

# install related package
pip install ultralytics
```

#### 1. Data format

Place the image in ./traffic_sign_dataset
```
./traffic_sign_dataset
├── images
│   ├── train
│   │   ├── train_image1.jpg
│   │   ├── train_image2.jpg
│   │   ├── ⋮
│   │   └── train_imagen.jpg
│   ├── val
│   │   ├── val_image1.jpg
│   │   ├── val_image2.jpg
│   │   ├── ⋮
│   │   └── val_imagen.jpg
│   └── test
│       ├── test_image1.jpg
│       ├── test_image2.jpg
│       ├── ⋮
│       └── test_imagen.jpg
└── labels
    ├── train
    │   ├── train_image1.txt
    │   ├── train_image2.txt
    │   ├── ⋮
    │   └── train_imagen.txt
    ├── val
    │   ├── val_image1.txt
    │   ├── val_image2.txt
    │   ├── ⋮
    │   └── val_imagen.txt
    └── test
        ├── test_image1.txt
        ├── test_image2.txt
        ├── ⋮
        └── test_imagen.txt
```



#### 2. Inference 

To generate the prediction outcome of the CDefficient model, 
```
python inference.py
```

After inference, the output directory structure will be:

```
./inference_result
└── traffic_best
    └── labels
        ├── test_image1.txt
        ├── test_image1.txt
        ├── ⋮
        └── test_imagen.txt

```


## Training
#### Training from scratch

Run this code in the terminal to train:
```
python train.py
```

#### Training

After training, the output directory structure will be:
```
./trained_model
└── CDefficient
    └── weights
        ├── best.pt
        └── last.pt

```


## License
This Python source code is released under a creative commons license, which allows for personal and research use only. For a commercial license please contact Prof Ching-Wei Wang. You can view a license summary here:  
http://creativecommons.org/licenses/by-nc/4.0/


## Contact
Prof. Ching-Wei Wang  
  
cweiwang@mail.ntust.edu.tw; cwwang1979@gmail.com  
  
National Taiwan University of Science and Technology

