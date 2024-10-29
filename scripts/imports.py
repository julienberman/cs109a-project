# Data Handling and Processing
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
import re
import os
import time

# Plotting
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# OCR and Image Processing
import cv2
import pytesseract
import easyocr
from pdf2image import convert_from_path

# Parallel Processing
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Machine Learning
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

# SSL (for handling certificate validation)
import ssl

from pathlib import Path
from pdf2image import convert_from_path