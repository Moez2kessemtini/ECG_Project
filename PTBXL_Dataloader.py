import pandas as pd
import numpy as np
import wfdb
import ast
import torch
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.pyplot as plt
import scipy.signal


def low_pass_filter(data, cutoff=20, fs=100, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    if not 0 < normal_cutoff < 1:
        raise ValueError("La fréquence critique normalisée doit être entre 0 et 1.")
    b, a = scipy.signal.butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = scipy.signal.filtfilt(b, a, data, axis=0)
    return filtered_data


class PTBXLDataLoader(Dataset):
    def __init__(self, csv_path, data_path, folds, sampling_rate=100, modality='all',
                 extract_heartbeats=True, apply_filter=False):
        self.data_path = data_path
        self.modality = modality
        self.sampling_rate = sampling_rate
        self.apply_filter = apply_filter
        self.extract_heartbeats = extract_heartbeats

        self.df = pd.read_csv(csv_path)
        self.df.scp_codes = self.df.scp_codes.apply(lambda x: ast.literal_eval(x))

        agg_df = pd.read_csv(os.path.join(data_path, 'scp_statements.csv'), index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]

        def aggregate_diagnostic(y_dic):
            tmp = [agg_df.loc[key].diagnostic_class for key in y_dic.keys() if key in agg_df.index]
            return list(set(tmp))

        self.df['diagnostic_superclass'] = self.df.scp_codes.apply(aggregate_diagnostic)
        self.df = self.df[self.df.strat_fold.isin(folds)]

        self.record_names, self.data_dicts = self.organize_data()

    def get_file_paths(self, df, sampling_rate, path):
        file_names = df.filename_lr if sampling_rate == 100 else df.filename_hr
        return [os.path.join(path, f) for f in file_names]

    def load_raw_data(self, file_path):
        signal, _ = wfdb.rdsamp(file_path)
        if self.apply_filter:
            signal = low_pass_filter(signal, cutoff=20, fs=self.sampling_rate, order=5)
        return signal

    def segment_by_heartbeat(self, leads_data):
        heartbeats_by_lead = {}
        total_heartbeats = 0

        for lead_idx, lead_data in enumerate(leads_data.T):  # Transpose to iterate over leads
            r_peaks, _ = scipy.signal.find_peaks(lead_data, distance=30, height=0.2)
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

            heartbeats_by_lead[lead_idx] = heartbeats
            total_heartbeats += len(heartbeats)

        return heartbeats_by_lead, total_heartbeats

    def organize_data(self):
        record_names = []
        data_dicts = []

        for _, row in self.df.iterrows():
            record_name = row.filename_lr if self.sampling_rate == 100 else row.filename_hr
            record_names.append(record_name)

            file_path = os.path.join(self.data_path, record_name)
            signal = self.load_raw_data(file_path)

            heartbeats_by_lead, total_heartbeats = self.segment_by_heartbeat(signal)

            data_dict = {i: heartbeats_by_lead.get(i, []) for i in range(12)}
            data_dict["Annotation"] = row.diagnostic_superclass
            data_dict["Total_heartbeats"] = total_heartbeats / 12

            data_dicts.append(data_dict)

        return record_names, data_dicts

    def __len__(self):
        return len(self.record_names)

    def __getitem__(self, idx):
        if self.extract_heartbeats:
            if self.modality == 'all':
                return self.data_dicts[idx]
            else:
                lead_idx = int(self.modality)
                return self.data_dicts[idx][lead_idx]
        else:
            file_path = self.get_file_paths(self.df, self.sampling_rate, self.data_path)[idx]
            data = self.load_raw_data(file_path)
            if self.modality == 'all':
                return data
            else:
                lead_idx = int(self.modality)
                return data[lead_idx]

    def visualize_data(self, signal, lead=None):
        if lead is None:
            num_leads = signal.shape[1]
            fig, axs = plt.subplots(num_leads, figsize=(15, num_leads * 2))
            for i in range(num_leads):
                axs[i].plot(signal[:, i])
                axs[i].set_title(f'Lead {i + 1}')
        else:
            fig, ax = plt.subplots(figsize=(15, 4))
            ax.plot(signal)
            ax.set_title(f'Lead {lead + 1}')
        plt.show()


if __name__ == "__main__":
    csv_path = r'C:\Users\User\Desktop\Projet stage\Second data ptb-xl\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\ptbxl_database.csv'
    data_path = r'C:\Users\User\Desktop\Projet stage\Second data ptb-xl\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3\\'

    all_folds = list(range(1, 11))
    dataset = PTBXLDataLoader(csv_path, data_path, all_folds, modality='all', apply_filter=True,
                              extract_heartbeats=True)
    data_dict = dataset.__getitem__(4444)
