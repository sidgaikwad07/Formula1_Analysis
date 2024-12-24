# Use the official Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Update the package list and install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python libraries for machine learning, deep learning, and data analysis
RUN pip install --no-cache-dir \
    notebook \
    pandas \
    numpy \
    matplotlib \
    seaborn \
    scikit-learn \
    scipy \
    statsmodels \
    torch \
    torchvision \
    tensorflow \
    xgboost \
    plotly \
    seaborn \
    joblib

# Expose the default Jupyter Notebook port
EXPOSE 8888

# Command to run the Jupyter Notebook server
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]

