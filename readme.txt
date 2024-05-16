Please follow these setup instructions carefully to ensure a smooth operation of the code.

Step 1: Upload Kaggle API Token
Before running the code, you need to upload kaggle.json file. This file contains your API credentials for Kaggle to download dataset.

This folder also provided a kaggle.json file.

Location: Upload the file to the content directory in your Colab environment, which is the default working directory.

Step 2: Adjust GPU Settings
If you are using the default T4 GPU in Colab, you need to adjust the memory settings to avoid exceeding GPU memory limits.

Batch Size: The current batch size is set to 450, occupying about 20GB of GPU memory. Reduce all batch sizes to prevent memory overflow.
Alternative GPU: If available, consider using an A100 GPU, which can handle the current settings without adjustments.

Step 3: Execute Notebook Cells
Execute each cell in the notebook in the order they appear. This sequential execution is crucial for maintaining the correct flow of operations and dependencies.