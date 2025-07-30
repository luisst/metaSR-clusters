
import pickle
from collections import Counter
from pathlib import Path
from comparison_framework_utils import define_key_sample_coordinates,\
        filter_and_plot,\
        plot_distance_histograms
from utils_luis import calculate_X_centroids

# run_ID
dataset_name = 'TestAO-Irmadb'
run_ID = 'EXP010C_TestAO-Irmadb_umap1H9_pca0_mcs25_ms5_eom'
exp_id = 'STG3_EXP010C-SHAS-DV-umap1H9'
mel_feats_id = 'STG2_EXP010C-SHAS-DV_feats'
feats_run_id = f'{dataset_name}_{run_ID}'
mel_feats_name = 'TestAO-Irmadb_SHAS_DV_feats'

current_run_id = f'{dataset_name}_{run_ID}_HDBSCAN'


root_dir = Path.home().joinpath('Dropbox','DATASETS_AUDIO', 'Proposal_runs')
exp_dir =  root_dir / dataset_name
prediction_dir = exp_dir / 'STG_3' / exp_id / 'HDBSCAN_pred_output'
mel_feats_dir = exp_dir / 'STG_2' / mel_feats_id 

path_pred_info = prediction_dir / f'{feats_run_id}_predinfo.pickle'
path_mel_feats = mel_feats_dir / f'{mel_feats_name}.pickle' 

output_folder_path = prediction_dir / f'{run_ID}_details'
if not output_folder_path.exists():
    output_folder_path.mkdir(parents=True, exist_ok=True)

with open(path_pred_info, 'rb') as handle:
    pred_labels, coordinates, audio_files = pickle.load(handle)

with open(path_mel_feats, "rb") as file:
    X_data_and_labels = pickle.load(file)
Mixed_X_data, Mixed_X_paths, Mixed_y_labels = X_data_and_labels

#audio_files_paths = process_paths(audio_files)

# Count occurrences of each label
label_counts = Counter(pred_labels)

# User input for label selection
label_prompt = "Available labels:\n"
for label, count in label_counts.items():
    label_prompt += f"  - Label {label}: {count} items\n"
label_prompt += "Enter a label to filter: "

selected_label = int(input(label_prompt))
if selected_label in label_counts:

    # Get the coordinates of the key sample for the selected label by averaging the coordinates of same label samples
    key_sample_coords = define_key_sample_coordinates(selected_label, coordinates, pred_labels)

    # Filter and plot    
    filter_and_plot(selected_label, audio_files, key_sample_coords, coordinates, pred_labels)
else:
    print("Invalid label selected.")


# Divide the X_train data in a dictionary with the labels as keys
centroids_dict = calculate_X_centroids(Mixed_X_data, pred_labels)
lbl_distances = plot_distance_histograms(centroids_dict, Mixed_X_data, pred_labels, output_folder_path, current_run_id)


# Store the centroid_dict
with open(f'{output_folder_path}/{current_run_id}_CM-dist.pickle', 'wb') as handle:
    pickle.dump(lbl_distances, handle, protocol=pickle.HIGHEST_PROTOCOL)