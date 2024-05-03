# ArtForgeriesDetection

## Introduction
This repository contains the code for our deep learning project focused on art authentication. The project aims to distinguish genuine artworks from forgeries related to Vincent van Gogh. We leverage convolutional neural networks (CNNs) with convolutional attention mechanism and contrastive learning to develop robust models capable of detecting subtle nuances between authentic paintings and forgeries.

## Installation
To replicate the environment and run the code follow these steps (**Warning**: a CUDA device must be used):

1. Clone the repository to your local machine 

    ```bash
    git clone https://github.com/alessandroAbati/ArtForgeriesDetection
    cd ArtForgeriesDetection
    ```

2. Install dependencies

    - Using conda
        ```bash
        conda env create -f environment.yml
        conda activate <environment_name>
        ```
        * Note: the default environemnt name will be 'ArtForgeriesDetection'; if you wish to modify the name, change the first line fo the environment.yml file before running.

    - Using pip
        ```bash
        pip install -r requirements.txt
      
    - WandB:
   
         To setup WandB logging, you must set the WandB key as an environment variable

## Usage

1. Multiclass fine-tuning

    - Dataset

        The dataset for the first multiclass fine-tuning of the models had been created by refining the Hugging-Face WikiArt dataset (https://huggingface.co/datasets/huggan/wikiart). The dataset we used is available here:
        https://drive.google.com/drive/folders/15nQBkj3PtQIqIG5EB2ECS66KMATazQRY?usp=sharing

        Please place the 'wikiart_data_batches' directory in the main 'ArtForgeriesDetection' folder. If you wish to change the path of the dataset directory, change the config.yaml file accordingly.

    - Adjust the settings in the 'config.yaml' file. 

        For example, to train 'EfficinetNet' model without attention the config file should look like this:
        ```yaml
        data_settings:
            dataset_path: 'wikiart_data_batches/data_batches_filtered'
            binary: False
            contrastive: False
            contrastive_batch_size: 24 # The batch will always be formed by 1 anchor, 3 positives, (batch_size-4) negatives
        model:
            model_type: 'efficientnet' # resnet or efficientnet or efficientnetAttention
            num_classes:  10 # If data_settings[binary] is True, number of classes will be always 2
            checkpoint_folder: 'checkpoints'
        train:
            epochs: 20
            batch_size: 8
            learning_rate: 0.00002
        ```

    - Train
        To start the training run the 'train.py' file:
        ```bash
        python train.py
        ```

2. Binary Classification

    - Adjust the settings in the 'config.yaml' file.

        For example, to train 'EfficinetNet' model without attention the config file should look like this:
        ```yaml
        data_settings:
            dataset_path: 'wikiart_data_batches/data_batches_filtered'
            binary: True
            contrastive: False
            contrastive_batch_size: 24 # The batch will always be formed by 1 anchor, 3 positives, (batch_size-4) negatives
        model:
            model_type: 'efficientnet' # resnet or efficientnet or efficientnetAttention
            num_classes:  10 # If data_settings[binary] is True, number of classes will be always 2
            checkpoint_folder: 'checkpoints'
        train:
            epochs: 20
            batch_size: 8
            learning_rate: 0.00002
        ```

        * Note: if you want to run the binary classification in contrastive mode, change 'data_settings[contrastive]' to True

    - Train

        To start the training run the 'train.py' file:
        ```bash
        python train.py
        ```

3. Visualisation of Latent Space Representation

    - Change the checkpoint file name to be loaded in the 'style.py' file at line 70 for the encoder and at line 75 for the head:

        The checkpoint files for the encoder has the format: '<model_name>\_<binary/multiclass>\_', while the checkpoint files for the head has the format: '<model_name>\_head\_<binary/multiclass>\_'
        * Note: the checkpoints between the encoder and the head MUST be consistent.

    - Visualise the plots:

        Run:
        ```bash
        python style.py
        ```

4. Inference and Visualisation of Attention

    - Change the checkpoint file name to be loaded in the 'inference.py' file at line 116 for the encoder and at line 120 for the head:

        The checkpoint files for the encoder has the format: '<model_name>\_<binary/multiclass>\_', while the checkpoint files for the head has the format: '<model_name>\_head\_<binary/multiclass>\_'
        * Note: the checkpoints between the encoder and the head MUST be consistent.

    - Visualisation of attention settings:
      
      By default, the attention matrix/weights are visualized. If the outputs of the attention model want to be visualized, change in 'models.py', in line 279, 'attention' variable for 'output'.
      Additionally, the line 114 in 'inference.py' must be set as 'attention=True' if attentions want to be plotted. Otherwise set this variable as False.

    - Visualize errors:
      
      If you want to visualize the errors made by the code, set the boolean variable 'plot_errors' on line 169 to True.

    - Single vs Multihead attention:
   
      If you want to compare singlhead vs multihead attention results, please swap line 162 for line 161 in 'models.py'.
  - Visualise the results:

      Run:
      ```bash
      python style.py
      ```