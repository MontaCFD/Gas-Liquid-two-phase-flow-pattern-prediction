# Project Setup Guide

## 1. Create Virtual Environment
python --version
Python 3.10.3
Run the following command to create a virtual environment for the project:
py -3.10 -m venv venv

## 2. Activate Virtual Environment
Navigate to the scripts directory: cd venv\scripts
Run the activation script: .\activate

## 3. Install Required Packages
With the virtual environment activated, install the required dependencies:
pip install -r requirements.txt

## 4.Generate requirements.txt
To update requirements.txt, run:
pip freeze > requirements.txt

## 5.Important Notes
> Always activate your virtual environment before running commands or scripts.
> Verify that Python 3.10 is properly installed and your virtual environment is correctly set up.
  python --version
  Python 3.10.3
> py -3.10 .\flow_map_mechanistic_machine_learning.py to ensure using python 3.10 while executing scripts
> you might need to install latex (Texlive or Miketex)
# Directory Structure
project-root/
├── venv/
├── data/
├── notebooks/
├── src/  
├── test/  
├── requirements.txt 
└── README.md 




