import os
import glob
import wfdb
from torch.utils.data import Dataset
import scipy.io
import matplotlib.pyplot as plt
import scipy.signal
import numpy as np

SNOMED_CT = {
    "1AVB": "270492004",
    "2AVB": "195042002",
    "2AVB1": "54016002",
    "2AVB2": "28189009",
    "3AVB": "27885002",
    "ABI": "251173003",
    "ALS": "39732003",
    "APB": "284470004",
    "AQW": "164917005",
    "ARS": "47665007",
    "AVB": "233917008",
    "CCR": "251199005",
    "CR": "251198002",
    "ERV": "428417006",
    "FQRS": "164942001",
    "IDC": "698252002",
    "IVB": "698252002",
    "JEB": "426995002",
    "JPT": "251164006",
    "LBBB": "164909002",
    "LBBBB": "164909002",
    "LFBBB": "164909002",
    "LVH": "164873001",
    "LVQRSAL": "251146004",
    "LVQRSCL": "251148003",
    "LVQRSLL": "251147008",
    "MI": "164865005",
    "MIBW": "164865005",
    "MIFW": "164865005",
    "MILW": "164865005",
    "MISW": "164865005",
    "PRIE": "164947007",
    "PWC": "164912004",
    "QTIE": "111975006",
    "RAH": "446358003",
    "RBBB": "59118001",
    "RVH": "89792004",
    "STDD": "429622005",
    "STE": "164930006",
    "STTC": "428750005",
    "STTU": "164931005",
    "TWC": "164934002",
    "TWO": "59931005",
    "UW": "164937009",
    "VB": "11157007",
    "VEB": "75532003",
    "VFW": "13640000",
    "VPB": "17338001",
    "VPE": "195060002",
    "VET": "251180001",
    "WAVN": "195101003",
    "WPW": "74390002",
    "SB": "426177001",
    "SR": "426783006",
    "AFIB": "164889003",
    "ST": "427084000",
    "AF": "164890007",
    "SA": "427393009",
    "SVT": "426761007",
    "AT": "713422000",
    "AVNRT": "233896004",
    "AVRT": "233897008",
    "SAAWR": "195101003",
}


class ECG12LeadDataset(Dataset):
    def __init__(self, path, modality='all', apply_filter=False, extract_heartbeats=True):
        self.records = get_list_file(os.path.expanduser(path))
        self.current_idx = None
        self.modality = modality
        self.apply_filter = apply_filter
        self.extract_heartbeats = extract_heartbeats

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        self.current_idx = idx
        record_name = self.records[idx]
        print(record_name)

        if self.extract_heartbeats:
            if self.modality == 'all':
                annotations = self.load_annotation(record_name)
                acronyms = self.extract_acronym(annotations)
                data = load_data(record_name)
                heartbeats = self.segment_by_heartbeat(data)
                return heartbeats, acronyms
            else:
                annotations = self.load_annotation(record_name)
                acronyms = self.extract_acronym(annotations)
                lead_idx = int(self.modality)
                data = self.choice_lead(record_name, lead_idx)
                heartbeats = self.segment_by_heartbeat(data)
                return heartbeats, data

        else:
            if self.modality == 'all':
                data = load_data(record_name)
                annotations = self.load_annotation(record_name)
                acronyms = self.extract_acronym(annotations)
                return data, acronyms
            else:
                lead_idx = int(self.modality)
                data = self.choice_lead(record_name,lead_idx)
                annotations = self.load_annotation(record_name)
                acronyms = self.extract_acronym(annotations)
                return data, acronyms


    def choice_lead(self, record_name, lead_number):
        leads_data = load_data(record_name, self.apply_filter)
        lead_data = leads_data[lead_number]
        return lead_data

    def load_annotation(self, record_name):
        hea_file = os.path.join(os.path.dirname(self.records[self.current_idx]), record_name)
        with open(hea_file, "r") as f:
            for line in f:
                if line.startswith('#Dx'):
                    annotation = line.strip().split(': ')[1]
                    anno_list = [int(ele) for ele in annotation.split(',')]
                    print(anno_list)
                    return anno_list

    def extract_acronym(self, annotations):
        final = []
        for elem in annotations:
            for k, v in SNOMED_CT.items():
                if str(elem) == v:
                    final.append(k)
        print(final)
        return final

    def segment_by_heartbeat(self, leads_data):
        if isinstance(leads_data, list):
            lead_data = leads_data[0]
        else:
            lead_data = leads_data

        # Détection des pics R
        r_peaks, _ = scipy.signal.find_peaks(lead_data, distance=150, height=300)

        num_heartbeats = len(r_peaks)
        heartbeats = []
        before = 70
        after = 200

        for i in range(len(r_peaks)):
            start = r_peaks[i] - before
            if start < 0:
                continue
            end = r_peaks[i] + after
            if end > len(lead_data):
                break
            heartbeat = lead_data[start:end]
            heartbeats.append(heartbeat)

        return heartbeats

    def get_heartbeat_data(self, leads_data):
        record_name = os.path.basename(self.records[self.current_idx]).split('.hea')[0]
        hea_file = os.path.join(os.path.dirname(self.records[self.current_idx]), record_name + '.hea')
        annotation = self.load_annotation(hea_file)

        heartbeats_dict = {}
        for i, lead_data in enumerate(leads_data):
            heartbeats, _ = self.segment_by_heartbeat(lead_data)
            heartbeats_dict[i] = heartbeats

        return record_name, heartbeats_dict, annotation


def get_list_file(path, extension='.hea'):
    print(path)
    pattern = path + '/*/*/*' + extension
    list_files = glob.glob(pattern)
    return list_files


def low_pass_filter(data, cutoff=50, fs=500, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = scipy.signal.butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = scipy.signal.filtfilt(b, a, data)
    return filtered_data


def load_data(record_name, apply_filter=True):
    mat_file = os.path.join(os.path.expanduser(record_name.split('.hea')[0]) + ".mat")
    mat_contents = scipy.io.loadmat(mat_file)
    val = mat_contents['val']
    print("val", val)
    plt.plot(np.transpose(val))
    plt.show()
    num_leads, num_samples = val.shape
    print(f"Nombre de leads : {num_leads}, Nombre d'échantillons par lead : {num_samples}")
    leads_data = [val[i, :] for i in range(12)]
    if apply_filter:
        leads_data = [low_pass_filter(lead) for lead in leads_data]
    return leads_data


def visualize_lead(lead_data):
    plt.figure(figsize=(10, 4))
    plt.plot(lead_data)
    plt.title("ECG Lead Data")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    path = r'C:\Users\User\Downloads\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0\a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0\WFDBRecords'
    base = ECG12LeadDataset(path, modality='2', apply_filter=True, extract_heartbeats=False)
    heartbeats = base.__getitem__(100)
