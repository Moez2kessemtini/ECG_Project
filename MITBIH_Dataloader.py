import os

import numpy as np
import wfdb
import matplotlib.pyplot as plt
import scipy.signal
from torch.utils.data import Dataset
from numpy import random

base_directory = r'C:/Users/User/Downloads/mit-bih-arrhythmia-database-1.0.0/mit-bih-arrhythmia-database-1.0.0'


def low_pass_filter(data, cutoff=20, fs=360, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = scipy.signal.butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = scipy.signal.filtfilt(b, a, data, axis=0)
    return filtered_data


class MITBIHDataset(Dataset):
    def __init__(self, base_dir, number_of_heartbeats, modality='all', extract_heartbeats=True, apply_filter=True):
        self.base_dir = base_dir
        self.modality = modality
        self.apply_filter = apply_filter
        self.extract_heartbeats = extract_heartbeats
        self.sampling_rate = 360
        self.sequences = self.find_sequences()
        self.number_of_heartbeats = number_of_heartbeats

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence_name = self.sequences[idx]
        patient_number = int(sequence_name.split('-')[0])

        if self.extract_heartbeats:
            if self.modality == 'all':
                heartbeats, labels = self.segment_by_heartbeat3(sequence_name)
                return heartbeats, labels
            else:
                lead_idx = int(self.modality)
                heartbeats, labels = self.segment_by_heartbeat3(sequence_name)
                if lead_idx == 0:
                    heartbeats_lead0 = heartbeats[0]
                    labels_0 = labels[0]
                    print(heartbeats_lead0.shape)
                    print(labels_0.shape)
                    return heartbeats_lead0, labels_0
                if lead_idx == 1:
                    heartbeats_lead1 = heartbeats[1]
                    labels_1 = labels[1]
                    return heartbeats_lead1, labels_1

        else:
            if self.modality == 'all':
                data = self.load_all(sequence_name)
                annotation = self.load_annotations(sequence_name)
                print("annotation symbols", annotation[0][1:-1])
                print("longeur annotation symbols", len(annotation[0][1:-1]))
                print("annotation samples", annotation[1][1:-1])
                heartbeat = self.segment_by_heartbeat3(sequence_name)
                print("heartbeat ", heartbeat)
                return heartbeat
            else:
                lead_idx = int(self.modality)
                lead_data = self.choice_lead(sequence_name, lead_idx)
                annotation = self.load_annotations(sequence_name)

                return lead_data, annotation

    def find_sequences(self):
        sequences = []
        for filename in os.listdir(self.base_dir):
            if filename.endswith(".hea"):
                sequence_name = os.path.splitext(filename)[0]
                sequences.append(sequence_name)
        return sequences

    def load_all(self, sequence_name):
        dat_file = os.path.join(self.base_dir, sequence_name)
        print(dat_file)
        record = wfdb.rdsamp(str(dat_file))
        data = record[0]
        print("data", data)
        print("longeur data", len(data))
        record_length = 2 * self.sampling_rate * 5

        if self.apply_filter:
            lead1_data_filtered = low_pass_filter(data[:, 0], cutoff=20, fs=self.sampling_rate)
            lead2_data = low_pass_filter(data[:, 1], cutoff=20, fs=self.sampling_rate)

        return data

    def choice_lead(self, sequence_name, lead_idx):
        dat_file = os.path.join(self.base_dir, sequence_name)
        record = wfdb.rdsamp(str(dat_file))
        data = record[0]
        print("dat_file", dat_file)
        record_length = 2 * self.sampling_rate * 5
        if lead_idx == 0:
            lead_data = data[:, 0]
        elif lead_idx == 1:
            lead_data = data[:, 1]

        if self.apply_filter:
            lead_data = low_pass_filter(lead_data, cutoff=20, fs=self.sampling_rate)
        plt.plot(lead_data)
        plt.show()
        return lead_data

    def load_annotations(self, sequence_name):
        annotation = wfdb.rdann(os.path.join(self.base_dir, sequence_name), extension='atr')
        mit_bih_labels_str = annotation.symbol

        labels_locations = annotation.sample

        return mit_bih_labels_str, labels_locations

    def segment_by_heartbeat3(self, sequence_name):
        mit_bih_labels_str, labels_locations = self.load_annotations(sequence_name)
        mit_bih_labels_str = mit_bih_labels_str[1:-1]
        print(mit_bih_labels_str)
        labels_locations = labels_locations[1:-1]
        data = self.load_all(sequence_name)
        heartbeats_lead1 = []
        heartbeats_lead2 = []
        all_heartbeats = []
        labels = []
        x = random.randint(len(labels_locations), size=(self.number_of_heartbeats))
        print(x)
        for i in x:
            heartbeat = data[labels_locations[i] - 70: labels_locations[i] + 200]
            heartbeat = np.transpose(heartbeat)
            heartbeat_1 = heartbeat[0]
            heartbeat_2 = heartbeat[1]
            heartbeats_lead1.append(heartbeat_1)
            heartbeats_lead2.append(heartbeat_2)
            lead_class = mit_bih_labels_str[i]
            labels.append(lead_class)
            all_heartbeats = np.vstack((heartbeats_lead1, heartbeats_lead2))
        print(all_heartbeats.reshape(2, int(all_heartbeats.shape[0] / 2), 270).shape)
        labels = np.vstack((labels, labels))
        print(labels.shape)

        return all_heartbeats.reshape(2, self.number_of_heartbeats, 270), labels

    def visualize_data(self, data, lead=None):
        print(data['lead1'].shape)
        if isinstance(data, dict):
            if 'lead1' in data and 'lead2' in data:
                plt.figure(figsize=(15, 6))
                plt.subplot(2, 1, 1)
                plt.plot(data['lead1'])
                plt.title('Lead 1')
                plt.subplot(2, 1, 2)
                plt.plot(data['lead2'])
                plt.title('Lead 2')
            elif 'lead' in data:
                plt.figure(figsize=(15, 4))
                plt.plot(data['lead'])
                plt.title(f'Lead {lead + 1}')
        plt.show()


if __name__ == "__main__":
    base = MITBIHDataset(
        base_directory,
        number_of_heartbeats=7,
        modality='all',
        extract_heartbeats=True,
        apply_filter=True
    )
    heartbeats = base.__getitem__(1)
