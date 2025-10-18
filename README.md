# 🩺 ECG Signal Processing & Dataset Dataloaders

### A Professional PyTorch-based ECG Dataset Loader Suite  

---

## 📖 Overview

This repository provides **modular and reusable dataloaders** for three of the most popular ECG datasets used in arrhythmia and cardiac disease research:  

- 🧩 **A Large-Scale 12-Lead ECG Database for Arrhythmia Study**  
- ❤️ **MIT-BIH Arrhythmia Database**  
- ⚡ **PTB-XL ECG Dataset**

Each dataloader is implemented as a **PyTorch `Dataset` class**, providing a consistent API for loading, filtering, segmenting, and visualizing ECG signals.  
This suite facilitates **ECG signal preprocessing, heartbeat segmentation, and diagnostic label extraction**, enabling researchers to train machine learning or deep learning models efficiently.

---

## 🧠 Key Features

✅ Unified PyTorch Dataset API for multiple ECG datasets  
✅ Automatic **heartbeat segmentation** using R-peak detection  
✅ Built-in **low-pass filtering** to reduce noise  
✅ Support for both **12-lead** and **2-lead** signals  
✅ Extraction of **diagnostic labels** (SNOMED-CT mappings for A_LARGE_SCALE)  
✅ **Visualization tools** for ECG signals and individual leads  
✅ Works with **raw WFDB** and **.mat** formats  

---

## 🏗️ Project Structure

```
ECG_Project/
│
├── A_LARGE_SCALE_Dataloader.py   # Dataloader for A Large-Scale 12-lead ECG Database
├── MITBIH_Dataloader.py          # Dataloader for MIT-BIH Arrhythmia Database
├── PTBXL_Dataloader.py           # Dataloader for PTB-XL Dataset
│
├── README.md                     # This file
└── requirements.txt              # (Optional) List of dependencies
```

---

## ⚙️ Installation

### 🐍 Requirements
You can install the required Python packages using:

```bash
pip install -r requirements.txt
```
## 📂 Usage

Each dataloader can be used independently depending on the dataset.

---

### 🔹 1. A_LARGE_SCALE ECG Dataset

```python
from A_LARGE_SCALE_Dataloader import ECG12LeadDataset

path = r'C:\path\to\a-large-scale-ecg-dataset\WFDBRecords'
dataset = ECG12LeadDataset(path, modality='2', apply_filter=True, extract_heartbeats=False)

data, labels = dataset[100]
```

**Features:**
- Extracts ECG signals from `.mat` files  
- Maps SNOMED-CT annotations to diagnostic acronyms  
- Optional low-pass filter  
- Optional heartbeat segmentation  

---

### 🔹 2. MIT-BIH Arrhythmia Database

```python
from MITBIH_Dataloader import MITBIHDataset

base_dir = r'C:\path\to\mit-bih-arrhythmia-database'
dataset = MITBIHDataset(base_dir, number_of_heartbeats=10, modality='all')

heartbeats, labels = dataset[1]
```

**Features:**
- Loads `.dat` and `.atr` files using WFDB  
- Segments beats automatically  
- Supports both leads and label synchronization  
- Optional noise filtering  

---

### 🔹 3. PTB-XL Dataset

```python
from PTBXL_Dataloader import PTBXLDataLoader

csv_path = r'C:\path\to\ptbxl_database.csv'
data_path = r'C:\path\to\ptbxl_dataset'

dataset = PTBXLDataLoader(csv_path, data_path, folds=[1, 2, 3], modality='all', apply_filter=True)
record = dataset[100]
```

**Features:**
- Integrates with PTB-XL metadata (`scp_codes`)  
- Extracts diagnostic superclass annotations  
- Performs heartbeat segmentation  
- Compatible with 100 Hz and 500 Hz versions  

---

## 🧪 Example Visualization

Each dataloader includes a visualization function:

```python
from A_LARGE_SCALE_Dataloader import visualize_lead

visualize_lead(lead_data)
```

Displays ECG signals with labeled axes for qualitative inspection.

---

## 🧩 Dataset References

- [A Large-Scale 12-Lead ECG Database for Arrhythmia Study](https://physionet.org/content/ecg-arrhythmia/1.0.0/)  
- [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/)  
- [PTB-XL ECG Dataset](https://physionet.org/content/ptb-xl/1.0.3/)

---

## 🧑‍💻 Author

**Moez Kessemtini**  
Final-Year Computer Engineering Student @ National Engineering School of Sfax (ENIS)  
📧 [kessemtinimoez2@gmail.com](mailto:kessemtinimoez2@gmail.com)  
🔗 [GitHub Profile](https://github.com/Moez2kessemtini)

---

