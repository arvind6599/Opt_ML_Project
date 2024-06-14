# OptML Project: On the Effect of Quantization on Deep Leakage from Gradients and Train-Test Accuracy

![](images/output_wo_quantization.png)

**Authors:** Arvind Satish Menon, Lars C.P.M. Quaedvlieg, and Sachin Bhadang

## Important Links
- In-depth project description: https://github.com/epfml/OptML_course/blob/master/labs/mini-project/miniproject_description.pdf

## Objective

This project aims to explore advanced quantization techniques to improve privacy preservation in distributed machine 
learning systems without significantly degrading model performance.

## Project introduction

Distributed machine learning systems have adopted the practice of sharing gradients rather than raw data to enhance
privacy. However, these shared gradients can still be susceptible to privacy breaches through inversion attacks.
Previous studies like Deep Leakage from Gradients (DLG) have shown that while mechanisms such as noise addition and
gradient compression can mitigate these attacks, they often result in a substantial decrease in model accuracy. This
project builds upon these findings by investigating quantization methods as a potential solution to balance both
privacy preservation and model performance effectively.

**Please note that not all the code is the work of this project group**. We will use a basis provided by the DLG paper.
For an idea of this basis, please utilize [this repository](https://github.com/mit-han-lab/dlg). However, we also
merged these methods with our project contributions.

## Getting Started

Clone the project to your local machine.

### Requirements

Make sure to install an environment with Python 3.10.12 to match the version we used in our paper.

Locate the repository and run:
```sh
pip install -r requirements.txt
```


### Usage
The repository contains code describing the different methods of analysis done in our report. We study the algorithm Deep Leakage from gradients (DLG) and in order to gain further insight, conduct experiments to test the capacity of the algorithm to extract sensitive data from the publicly shared gradient vectors.


## Structure of the repository

    ├── main.ipynb              # Study the effect of quantization and compression on the model performance by measuring the train-test loss and also the test accuracy 
    ├── Deep_Leakage_from_Gradients.ipynb               # Tests the quality of reconstructed images from quantized and compressed gradient tensors
    ├── Deep_Leakage_from_Experiments.ipynb               # Study the claim that DLG works for all convergence status i.e. it can extract sensitive data from the gradients at any point during the training process
    ├── Gradient_Quantization.ipynb               # Simple illustration of the Deep Leakage algorithm on a dummy problem
    ├── Qualitative_Evaluation.ipynb               # Computing the average MSE values after running DLG on different seeds to test its convergence behaviour
    ├── Qualitative_Evaluation.ipynb               # Computing the average MSE values after running DLG on different seeds to test its convergence behaviour
    ├── .gitignore        # GitHub configuration settings
    ├── README.md         # Description of the repository
    └── requirements.txt  # Python packages to install before running the code

### Images

> This is the directory in which all the image files generated from the code are stored


### Checkpoints

> This is the directory in which all the checkpoint files containing the model weights and their corresponding gradient tensors from different phases (start, middle, near to convergence) of training are stored. 

### Authors 

- Arvind S. Menon 
- Lars Quaedvlieg 
- Sachin Bhadang 