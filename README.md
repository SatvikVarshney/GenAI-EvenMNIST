# GenAI-EvenMNIST

## Overview
The GenAI-EvenMNIST project uses a Variational Auto-Encoder (VAE) with convolutional layers to generate images of even digits from the MNIST dataset. By leveraging generative AI, this project aims to explore the capabilities of neural networks in synthesizing digit images that resemble handwritten examples. It focuses on the intricate process of learning data distributions and generating new, unseen images through the lens of deep learning.

## Features
- **Even MNIST Dataset Preparation**: Utilizes a modified version of the MNIST dataset, focusing exclusively on even digits, resized for efficient processing.
- **Convolutional VAE**: Implements a VAE with convolutional layers tailored for image data, capable of learning complex patterns in digit images.
- **Image Generation**: After training, the model can generate new digit images based on learned distributions, showcasing the potential of generative AI.
- **Performance Visualization**: Includes loss metrics visualization to assess the model's training progress and convergence.
- **Verbose Mode**: Offers iterative reports on the model's learning progress, enhancing transparency and understanding of the training process.

##Results

### Sample outputs of generated images 

#### These Images are generated based on hand written characters as stored in the MNIST number dataset

| 0 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|
| ![image](https://github.com/SatvikVarshney/GenAI-EvenMNIST/assets/114079530/e89df22c-9f01-44d1-b09b-d653731312e5)| ![image](https://github.com/SatvikVarshney/GenAI-EvenMNIST/assets/114079530/4cc51f0a-2441-4545-ad2d-79ca2ef07ac8)| ![image](https://github.com/SatvikVarshney/GenAI-EvenMNIST/assets/114079530/4a07606b-1d95-4035-9def-0e8086dc2a6e)| ![image](https://github.com/SatvikVarshney/GenAI-EvenMNIST/assets/114079530/cf79a932-6905-4915-9aaf-979062c1f33a)| ![image](https://github.com/SatvikVarshney/GenAI-EvenMNIST/assets/114079530/6a966d63-d2f4-45bc-8d65-14c9df40371a)|
| ![image](https://github.com/SatvikVarshney/GenAI-EvenMNIST/assets/114079530/e32f8c68-8ead-4181-a0ee-8c9fc8420792)| ![image](https://github.com/SatvikVarshney/GenAI-EvenMNIST/assets/114079530/5b20e3a4-e5e0-44a5-92b3-f6598c190559)| ![image](https://github.com/SatvikVarshney/GenAI-EvenMNIST/assets/114079530/7dfed9c4-9dc7-4996-bd2e-ad3a3c3b4339)| ![image](https://github.com/SatvikVarshney/GenAI-EvenMNIST/assets/114079530/22637c8f-49c1-4806-9844-ee22bc523c62)| ![image](https://github.com/SatvikVarshney/GenAI-EvenMNIST/assets/114079530/4d32785e-d971-407d-bc6b-bc84ed3d8874)|
| ![image](https://github.com/SatvikVarshney/GenAI-EvenMNIST/assets/114079530/64249b87-62ac-4d15-af1f-ef67bae4bedd)| ![image](https://github.com/SatvikVarshney/GenAI-EvenMNIST/assets/114079530/610c7614-35bd-4cb9-a97b-11891e4222e4)| ![image](https://github.com/SatvikVarshney/GenAI-EvenMNIST/assets/114079530/1a641d67-3fcf-4a8f-ab57-cbb80ebdd925)| ![image](https://github.com/SatvikVarshney/GenAI-EvenMNIST/assets/114079530/0c78a834-a875-4141-9c65-f159d13e7c7f)| ![image](https://github.com/SatvikVarshney/GenAI-EvenMNIST/assets/114079530/ab20ec22-4c80-457c-ad46-0d3097557d7c)|


### Training Loss Results

![Loss](https://github.com/SatvikVarshney/GenAI-EvenMNIST/assets/114079530/ec13a67b-792a-418f-9adc-e3edd973de6e)

The graph above illustrates the VAE's training loss over epochs. The model starts with a higher loss which quickly decreases, indicating effective learning. As the epochs progress, the loss continues to diminish, suggesting that the model is improving its ability to generate even MNIST digits with a consistent decline towards convergence.



## Getting Started

### Prerequisites
- Python 3.x
- PyTorch
- NumPy
- Matplotlib

### Data Files
The dataset, `even_mnist.csv`, consists of MNIST digit images that have been filtered to include only even numbers and resized to 14x14 pixels. It's located in the `data/` directory.

### Configuration Files
Model hyperparameters and operational parameters can be adjusted in a JSON configuration file. The configurations include learning rate, number of training epochs, batch size, and latent space dimensions.

Example `param.json`:

```json
{
	"learning rate": 0.001,
	"num iter": 5,
	"batch_size" : 128,
	"latent_dim": 64,
	"verbose_mode" : 1,
	"img_dim": 14
}
```


Certainly! Here's a README.md structure for your "GenAI-EvenMNIST" project, tailored to the structure you've provided:

markdown
Copy code
# GenAI-EvenMNIST

## Overview
The GenAI-EvenMNIST project uses a Variational Auto-Encoder (VAE) with convolutional layers to generate images of even digits from the MNIST dataset. By leveraging generative AI, this project aims to explore the capabilities of neural networks in synthesizing digit images that resemble handwritten examples. It focuses on the intricate process of learning data distributions and generating new, unseen images through the lens of deep learning.

## Features
- **Even MNIST Dataset Preparation**: Utilizes a modified version of the MNIST dataset, focusing exclusively on even digits, resized for efficient processing.
- **Convolutional VAE**: Implements a VAE with convolutional layers tailored for image data, capable of learning complex patterns in digit images.
- **Image Generation**: After training, the model can generate new digit images based on learned distributions, showcasing the potential of generative AI.
- **Performance Visualization**: Includes loss metrics visualization to assess the model's training progress and convergence.
- **Verbose Mode**: Offers iterative reports on the model's learning progress, enhancing transparency and understanding of the training process.

## Getting Started

### Prerequisites
- Python 3.x
- PyTorch
- NumPy
- Matplotlib

### Data Files
The dataset, `even_mnist.csv`, consists of MNIST digit images that have been filtered to include only even numbers and resized to 14x14 pixels. It's located in the `data/` directory.

### Configuration Files
Model hyperparameters and operational parameters can be adjusted in a JSON configuration file. The configurations include learning rate, number of training epochs, batch size, and latent space dimensions.

Example `param.json`:

```json
{
  "learning_rate": 0.001,
  "num_epochs": 50,
  "batch_size": 64,
  "latent_dim": 20,
  "image_size": 14,
  "verbose_mode": 1
}
```

### Installation
Clone the repository to get started:
```bash
git clone https://github.com/SatvikVarshney/GenAI-EvenMNIST.git
```

Navigate to the project directory:
```bash
cd GenAI-EvenMNIST
```

Install the required Python packages:
```bash
pip install -r requirements.txt
```

#### Command line syntax to run main.py
```
python main.py -o results -n 100
OR
python main.py --help
```

#### Command line parameters description
```
Parameter 1 = -o results
Parameter 1 description = Folder path to store the regerenated images and loss plot
Parameter 1 default value = results
Parameter 1 type = string

Parameter 2 = -n 100
Parameter 2 description = Count of the regenerated output files
Parameter 2 default value = 100
Parameter 3 type = interger

```


