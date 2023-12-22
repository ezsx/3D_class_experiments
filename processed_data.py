from tqdm import tqdm
import os
normal_scan_paths_0_100 = [
    os.path.join("/mnt/c/3D_class/MosMedData-CT-nifti/0_100_studies_nifti", x)
    for x in os.listdir(r"/mnt/c/3D_class/MosMedData-CT-nifti/0_100_studies_nifti")
]

normal_scan_paths_100_200 = [
    os.path.join("/mnt/c/3D_class/MosMedData-CT-nifti/100_200_studies_nifti", x)
    for x in os.listdir("/mnt/c/3D_class/MosMedData-CT-nifti/100_200_studies_nifti")
]

normal_scan_paths_200_300 = [
    os.path.join("/mnt/c/3D_class/MosMedData-CT-nifti/200_300_studies_nifti", x)
    for x in os.listdir("/mnt/c/3D_class/MosMedData-CT-nifti/200_300_studies_nifti")
]

normal_scan_paths_300_400 = [
    os.path.join("/mnt/c/3D_class/MosMedData-CT-nifti/300_400_studies_nifti", x)
    for x in os.listdir("/mnt/c/3D_class/MosMedData-CT-nifti/300_400_studies_nifti")
]

abnormal_scan_paths_400_500 = [
    os.path.join("/mnt/c/3D_class/MosMedData-CT-nifti/400_500_studies_nifti", x)
    for x in os.listdir("/mnt/c/3D_class/MosMedData-CT-nifti/400_500_studies_nifti")
]

abnormal_scan_paths_500_600 = [
    os.path.join("/mnt/c/3D_class/MosMedData-CT-nifti/500_600_studies_nifti", x)
    for x in os.listdir("/mnt/c/3D_class/MosMedData-CT-nifti/500_600_studies_nifti")
]

abnormal_scan_paths_600_700 = [
    os.path.join("/mnt/c/3D_class/MosMedData-CT-nifti/600_700_studies_nifti", x)
    for x in os.listdir("/mnt/c/3D_class/MosMedData-CT-nifti/600_700_studies_nifti")
]

abnormal_scan_paths_700_800 = [
    os.path.join("/mnt/c/3D_class/MosMedData-CT-nifti/700_800_studies_nifti", x)
    for x in os.listdir("/mnt/c/3D_class/MosMedData-CT-nifti/700_800_studies_nifti")
]

normal_scan_paths_all = normal_scan_paths_0_100 + normal_scan_paths_100_200 + normal_scan_paths_200_300 + normal_scan_paths_300_400
abnormal_scan_paths_all = abnormal_scan_paths_400_500 + abnormal_scan_paths_500_600 + abnormal_scan_paths_600_700 + abnormal_scan_paths_700_800

normal_scan_paths = []
abnormal_scan_paths = []

for dir in tqdm(normal_scan_paths_all, desc="Processing normal paths_all"):
    files = os.listdir(dir)
    nii_files = [file for file in files if file.endswith('.nii')]
    if len(nii_files) == 1:
        nii_file_path = os.path.join(dir, nii_files[0])
        normal_scan_paths.append(nii_file_path)

for dir in tqdm(abnormal_scan_paths_all, desc="Processing abnormal paths_all"):
    files = os.listdir(dir)
    nii_files = [file for file in files if file.endswith('.nii')]
    if len(nii_files) == 1:
        nii_file_path = os.path.join(dir, nii_files[0])
        abnormal_scan_paths.append(nii_file_path)

print(normal_scan_paths)
print(abnormal_scan_paths)
print("CT scans with normal lung tissue: " + str(len(normal_scan_paths)))
print("CT scans with abnormal lung tissue: " + str(len(abnormal_scan_paths)))
import skimage
import nibabel as nib
from scipy import ndimage
import dicom2nifti
import gzip
import shutil
from multiprocessing import Pool

def read_nifti_file(filepath):
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan


def normalize(volume):
    min = -1000
    max = 200
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume


def resize_volume(img):
    desired_depth = 64
    desired_width = 128
    desired_height = 128
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img

#
# def process_scan2(path):
#     """Read and resize volume"""
#     # Read scan
#     volume = read_nifti_file(path)
#     volume = normalize(volume)
#     volume = exposure.equalize_hist(volume)
#     volume = skimage.morphology.opening(volume, ball(2))
#     volume = gaussian(volume, sigma=2)
#     volume = resize_volume(volume)
#     return volume



import concurrent
import numpy as np
import pickle
import os
import gc
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import time

def process_scan(path):
    try:
        volume = read_nifti_file(path)
        volume = normalize(volume)
        volume = resize_volume(volume)
        return volume
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None

def save_scans(scans, label, save_dir="/mnt/c/3D_class/pickle_temp_stock/"):
    save_dir+=label
    os.makedirs(save_dir, exist_ok=True)
    timestamp = int(time.time())
    filename = f"{label}_{timestamp}.pickle"
    full_path = os.path.join(save_dir, filename)
    with open(full_path, 'wb') as file:
        pickle.dump(scans, file)
    print(f"Saved {len(scans)} scans to {full_path}")
    gc.collect()

def process_and_save_scans(scan_paths, label, max_workers=8):
    processed_scans = []
    batch_size = 10
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_scan, path) for path in scan_paths]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(scan_paths)):
            scan = future.result()
            if scan is not None:
                processed_scans.append(scan)
            if len(processed_scans) == batch_size:
                save_scans(processed_scans, label)
                processed_scans = []
                gc.collect()
        if processed_scans:  # Save remaining scans
            save_scans(processed_scans, label)

# Process and save scans
# print("Processing and Saving Abnormal Scans")
# process_and_save_scans(abnormal_scan_paths, 'abnormal')

print("Processing and Saving Normal Scans")
process_and_save_scans(normal_scan_paths, 'normal')


def load_scans_from_directory(directory):
    all_scans = []
    for filename in os.listdir(directory):
        if filename.endswith('.pickle'):
            full_path = os.path.join(directory, filename)
            with open(full_path, 'rb') as file:
                scans = pickle.load(file)
                all_scans.extend(scans)
    return all_scans

# Example usage
abnormal_scans = load_scans_from_directory('/mnt/c/3D_class/pickle_temp_stock/abnormal')
normal_scans = load_scans_from_directory('/mnt/c/3D_class/pickle_temp_stock/normal')
# Rest of the code remains the same

# Label assignment and dataset split
abnormal_labels = np.array([1 for _ in range(len(abnormal_scans))])
normal_labels = np.array([0 for _ in range(len(normal_scans))])

# Calculate indices for an 80/20 split
index_abnormal = int(0.8 * len(abnormal_scans))
index_normal = int(0.8 * len(normal_scans))

# Split data for training and validation
x_train = np.concatenate((abnormal_scans[:index_abnormal], normal_scans[:index_normal]), axis=0)
y_train = np.concatenate((abnormal_labels[:index_abnormal], normal_labels[:index_normal]), axis=0)
x_val = np.concatenate((abnormal_scans[index_abnormal:], normal_scans[index_normal:]), axis=0)
y_val = np.concatenate((abnormal_labels[index_abnormal:], normal_labels[index_normal:]), axis=0)

print(f"Number of samples in train and validation are {x_train.shape[0]} and {x_val.shape[0]}.")

# Save processed data
data_to_save = {
    'x_train': x_train,
    'y_train': y_train,
    'x_val': x_val,
    'y_val': y_val
}

with open('processed_data_stock.pickle', 'wb') as file:
    pickle.dump(data_to_save, file)

print("Data has been processed and saved.")