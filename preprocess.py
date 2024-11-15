import os
import mne
import json
import argparse
import numpy as np
from tqdm import tqdm
import src.config as config

mne.set_log_level('ERROR')

def parse():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--split_dir', type=str, default=str("data/splits"))
    parser.add_argument('--input_dir', type=str, default=str("pruned"))
    parser.add_argument('--output_dir', type=str, default=str("preprocessed"))
    parser.add_argument('--channels', type=str, default=None, help='Comma-separated list of channels to select, e.g., AF3,F3,FC5,Oz,O2')
    parser.add_argument('--bands', type=str, default='theta,alpha,beta,gamma,all', help='Comma-separated list of frequency bands to process, e.g., alpha,beta')

    args = parser.parse_args()
    
    # add data_directory
    args.data_dir = config.get_attribute("dataset_path")
    
    return args

class ResidualNan(Exception):
    pass

def interpolate(raw_data):
    
    # replace very large values with nans
    raw_data[abs(raw_data) > 1e2] = np.nan

    # get indices of nans
    nan_indices = np.where(np.isnan(raw_data))
    nan_indices = np.vstack(nan_indices).transpose()

    # hypotesis, Punctual nans
    for channel, timepoint in nan_indices:

        # get value before the point
        before = raw_data[channel, timepoint-1]
        # get value after the point
        after = raw_data[channel, timepoint-1]

        # interpolate
        raw_data[channel, timepoint] = (before + after) / 2

    nan_indices = np.where(np.isnan(raw_data))
    nan_indices = np.vstack(nan_indices).transpose()
    any_nan = nan_indices.shape[0]!=0
    if any_nan:
        raise ResidualNan("Data still contain Nans after interpolation")
        
    return raw_data

def open_and_interpolate(file):
    raw_file = mne.io.read_raw_fif(file, preload=True)
    raw_data = raw_file.get_data()
    try:
        raw_data = interpolate(raw_data)
    except ResidualNan as e:
        print(f"Residual NaNs in {file}")
        return None, None
    # Set the interpolated data back into raw_file
    raw_file._data = raw_data
    return raw_file, raw_data

def get_stats(file_list, args, band_name):
    l_freq, h_freq = bands[band_name]
    tmp = []
    for file in tqdm(file_list):
        raw_file, raw_data = open_and_interpolate(file)
        if raw_file is None:
            continue
        # Process raw_file
        if args.channels is not None:
            channels = args.channels.split(',')
            raw_file.pick_channels(channels)
        raw_file.set_eeg_reference('average', projection=False)
        raw_file.filter(4.0, 45.0, fir_design='firwin')
        raw_file.filter(l_freq, h_freq, fir_design='firwin')
        raw_data_band = raw_file.get_data()
        tmp.append(raw_data_band)
    if len(tmp) == 0:
        print(f"No data found for band {band_name}")
        return None, None
    # concatenate all the data
    data = np.concatenate(tmp, axis=1)
    # compute the mean and std
    mean = np.mean(data, axis=1)
    std = np.std(data, axis=1)
    return mean, std

def z_score(raw_data, mean, std):
    return (raw_data - mean[:, np.newaxis]) / std[:, np.newaxis]

def main(args):
    
    input_dir = os.path.join(args.data_dir, args.input_dir)
    output_dir = os.path.join(args.data_dir, args.output_dir)
    
    print(f"Input directory: {input_dir}")
    input_dirs = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]
    # create the same directory structure in the output directory
    for d in input_dirs:
        print(f"Creating directory {os.path.join(output_dir, os.path.basename(d))}")
        os.makedirs(os.path.join(output_dir, os.path.basename(d)), exist_ok=True)
        
    # Create a list of input files to process
    print("Listing files...")
    files = []
    for dir in input_dirs:
        files.extend([os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(".fif")])
    print(f"Found {len(files)} files to process!")
    
    print("Loading splits...")
    # Load the split file
    split_file_1 = os.path.join(args.split_dir, "splits_subject_identification.json")
    split_file_2 = os.path.join(args.split_dir, "splits_emotion_recognition.json")
    
    splits_1 = json.load(open(split_file_1, 'r'))
    splits_2 = json.load(open(split_file_2, 'r'))
    
    splits = {
        "train": splits_2["train"],
        "val_trial": splits_2["val_trial"],
        "val_subject": splits_2["val_subject"],
        "test_trial": splits_1["test_trial"],
        "test_subject": splits_2["test_subject"]

    }
    
    # Create a list with only train files for statistics
    train_files = [os.path.join(input_dir, "train", f"{s['id']}_eeg.fif") for s in splits["train"]]
    print(f"Found {len(train_files)} train files!")
    
    # Define frequency bands
    all_bands = {
        'theta': (4, 8),
        'alpha': (8, 15),
        'beta': (15, 32),
        'gamma': (32, 40),
        'all': (4, 40)
    }

    # Get bands to process from args
    bands_to_process = args.bands.split(',')

    # Filter the bands dictionary
    global bands
    bands = {band_name: all_bands[band_name] for band_name in bands_to_process if band_name in all_bands}
    
    # Get global train statistics (per channel) for each band
    mean_per_band = {}
    std_per_band = {}
    for band_name in bands.keys():
        print(f"Computing global statistics for band {band_name}...")
        mean, std = get_stats(train_files, args, band_name)
        mean_per_band[band_name] = mean
        std_per_band[band_name] = std
        print(f"Global statistics for band {band_name} computed!")
    
    # Uncomment the following lines if you want to compute subject-wise statistics
    '''
    print("Computing subject-wise statistics...")
    # Create a list with only train files for each subject for statistics
    train_files_per_subject = {}
    for file in splits["train"]:
        subject = file["subject_id"]
        if subject not in train_files_per_subject:
            train_files_per_subject[subject] = []
        train_files_per_subject[subject].append(os.path.join(input_dir, "train", f"{file['id']}_eeg.fif"))
    
    # Get train statistics per subject
    stats_per_subject = {}
    for subject_id, files in train_files_per_subject.items():
        mean_subj = {}
        std_subj = {}
        for band_name in bands.keys():
            print(f"Computing statistics for subject {subject_id}, band {band_name}...")
            mean, std = get_stats(files, args, band_name)
            mean_subj[band_name] = mean
            std_subj[band_name] = std
            print(f"Statistics for subject {subject_id}, band {band_name} computed!")
        stats_per_subject[subject_id] = {'mean': mean_subj, 'std': std_subj}
    print("Subject-wise statistics computed!")
    '''
    
    print("Preprocessing data...")
    # Process each file
    for file in tqdm(files):
        input_file = file
        output_file = file.replace(".fif", ".npy").replace(input_dir, output_dir)
        
        raw_file, raw_data = open_and_interpolate(input_file)
        if raw_file is None:
            continue
        # Process raw_file
        if args.channels is not None:
            channels = args.channels.split(',')
            raw_file.pick_channels(channels)
        raw_file.set_eeg_reference('average', projection=False)
        raw_file.filter(4.0, 45.0, fir_design='firwin')
        
        for band_name in bands.keys():
            raw_band = raw_file.copy()
            l_freq, h_freq = bands[band_name]
            raw_band.filter(l_freq, h_freq, fir_design='firwin')
            raw_data_band = raw_band.get_data()
            mean = mean_per_band[band_name]
            std = std_per_band[band_name]
            z_data = z_score(raw_data_band, mean, std)
            #output_file_band = output_file.replace('.npy', f'_{band_name}.npy')
            np.save(output_file, z_data)
    
    print("Preprocessing done!")
        
if __name__ == "__main__":
    args = parse()
    main(args)