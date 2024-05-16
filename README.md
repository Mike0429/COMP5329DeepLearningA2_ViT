# COMP5329DeepLearningA2_ViT

Welcome to the COMP5329 Deep Learning Assignment 2 project using Vision Transformer (ViT)! Please follow the steps below to set up your environment and prepare for running the code.

## Step 1: Create Virtual Environment and Activate

It's best practice to use a virtual environment to manage dependencies for your project. Here’s how you can create and activate one:

### For Windows
```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\activate
```
### For macOS and Linux
```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```
## Step 2: Install Required Packages
Download the necessary packages specified in the requirements.txt file located in the repository. Use pip to install these packages.

```bash
pip install -r requirements.txt
```

## Step 3: Configure CUDA and cuDNN
To leverage GPU acceleration, ensure that your CUDA and cuDNN environments are correctly configured. Follow the official CUDA Installation Guide and the cuDNN Installation Guide provided by NVIDIA.

Make sure to match the CUDA version with the version supported by your installed PyTorch.

## Step 4: Train the Model
Run the VisionTransformer_small.py script to train the Vision Transformer model.

```bash
python VisionTransformer_small.py
```

## Step 5: Predict with the Model
Run the predict.py script to make predictions using the trained model.

```bash
python predict.py
```

