# ArtForgeriesDetection

## Introduction
This repository contains the code for our deep learning project focused on art authentication. The project aims to distinguish genuine artworks from forgeries related to Vincent van Gogh. We leverage convolutional neural networks (CNNs) with convolutional attention mechanism and contrastive learning to develop robust models capable of detecting subtle nuances between authentic paintings and forgeries.

## Installation
To replicate the environment and run the code follow these steps:

1. Clone the repository to your local machine 
```bash
git clone https://github.com/alessandroAbati/ArtForgeriesDetection
cd ArtForgeriesDetection
```

2. Install dependancies

2.1 Using conda
```bash
conda env create -f environment.yml
conda activate <environment_name>
```
* Note: the default environemnt name will be 'ArtForgeriesDetection'; if you wish to modify the name, change the first line fo the environment.yml file before running.

2.2 Using pip
```bash
pip install -r requirements.txt
```