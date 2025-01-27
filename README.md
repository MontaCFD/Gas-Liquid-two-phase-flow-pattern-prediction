# Gas Liquid flow pattern prediction in horizontal and slightly inclined pipes.
This project presents a study to the prediction of flow pattern in horizontal an slightly inclined pipes using mechanistic model and machine learning. The results of this study are presented in this [paper](https://doi.org/10.1016/j.apm.2024.115748) published in [Applied mathematical modelling](https://www.sciencedirect.com/journal/applied-mathematical-modelling). If you use our code for your own research, we would be grateful if you cite our publication:
```bash
M. Guesmi, J. Manthey, S. Unz et al., Gas Liquid flow pattern prediction in horizontal and slightly inclined pipes: From mechanistic modelling to machine learning, *Applied Mathematical Modelling*, 115748, doi: [https://doi.org/10.1016/j.apm.2024.115748](https://doi.org/10.1016/j.apm.2024.115748).
```
 
 # Project Setup Guide

## 1. Create Virtual Environment

First, check your Python version to ensure you have Python 3.10 installed:

```bash
python --version
```
**Expected Output:**
```
Python 3.10.3
```

Run the following command to create a virtual environment for the project:

```bash
py -3.10 -m venv venv
```

## 2. Activate Virtual Environment

Run the activation script:

```bash
.\venv\Scripts\activate
```

## 3. Install Required Packages

With the virtual environment activated, install the required dependencies:

```bash
pip install -r requirements.txt
```

## 4. Generate `requirements.txt`

To update `requirements.txt` with the current packages installed in your virtual environment, run:

```bash
pip freeze > requirements.txt
```

## 5. Important Notes

- **Always activate your virtual environment** before running commands or scripts.
- Verify that Python 3.10 is properly installed and your virtual environment is correctly set up by running:
  
  ```bash
  python --version
  ```
  
  **Expected Output:**
  ```
  Python 3.10.3
  ```

- Use the following command to ensure you are using Python 3.10 while executing scripts:

  ```bash
  py -3.10 .\flow_map_mechanistic_machine_learning.py
  ```

- You might need to install LaTeX (Texlive or MikTeX) for plot generation.

## 6. Directory Structure

Here’s the structure of the project directory:

```
project-root/
├── venv/
├── data/
├── notebooks/
├── src/  
├── test/  
├── requirements.txt 
└── README.md
```
```
