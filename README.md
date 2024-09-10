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

Navigate to the `Scripts` directory:

```bash
cd venv\Scripts
```

Run the activation script:

```bash
.\activate
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
