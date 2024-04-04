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

#### Verbose mode output sample
```
Verbose mode = 0/OFF
Executing in Verbose Mode = 0
Processing Epoch 1/30....
Processing Epoch 2/30....
Processing Epoch 3/30....
...
...
...
Processing Epoch 29/30....
Processing Epoch 30/30....
Processing Sample regenerated images ....
********* Processing Completed *********

Verbose mode = 1/ON
Executing in Verbose Mode = 1
Processing Epoch 1/30....
Epoch 1 Training Loss = : 56.5530
Processing Epoch 2/30....
Epoch 2 Training Loss = : 43.8068
Processing Epoch 3/30....
Epoch 3 Training Loss = : 42.2278
...
...
...
Epoch 29 Training Loss = : 38.5104
Processing Epoch 30/30....
Epoch 30 Training Loss = : 38.4338
Processing Sample regenerated images ....
********* Processing Completed *********

```
