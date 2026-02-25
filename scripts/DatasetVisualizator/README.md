# Data Verification Tool

This tool provides a GUI for verifying and managing image datasets, particularly for handling image masks and quarantining invalid data. It was designed to be used with datasets organized in specific folders.

## Features
- Interactive GUI built with Tkinter.
- Displays images with overlaid masks using Matplotlib.
- Allows users to accept or reject images.
- Automatically moves rejected images and masks to a quarantine folder.
- Updates a blacklist CSV with rejected images.

## Requirements
- Python 3.x
- Tkinter
- PIL (Pillow)
- NumPy
- Matplotlib

## Installation
You can install the required dependencies using pip:
```bash
pip install -r requirements.txt
set DATABASE_VALKYRIE="path/to/databases/folder/container"
```

# Usage
To run the tool, execute the following command:
```bash
python main.py
```