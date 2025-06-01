
import pickle
from collections import Counter
from pathlib import Path
from comparison_framework_utils import define_key_sample_coordinates, filter_and_plot

# run_ID
run_ID = 'EXP010C_TestAO-Irmadb_umap1H9_pca0_mcs25_ms5_eom'
exp_id = 'STG3_EXP010C-SHAS-DV-umap1H9'

root_dir = Path.home().joinpath('Dropbox','DATASETS_AUDIO', 'Proposal_runs')
pickle_dir = root_dir.joinpath('TestAO-Irmadb', 'STG_3', exp_id, 'HDBSCAN_pred_output')

path_Xpaths = pickle_dir / f'{run_ID}_Xpaths.pickle'

path_predlbl = pickle_dir / f'{run_ID}_predlbl.pickle'

path_xtsn2d = pickle_dir / f'{run_ID}_xtsne2d.pickle'

# Give me the code to load Mixed_X_paths and Mixed_y_labels
with open(path_Xpaths, 'rb') as handle:
    audio_files = pickle.load(handle)

with open(path_predlbl, 'rb') as handle:
    labels = pickle.load(handle)

# Give me the code to load x_tsne_2d
with open(path_xtsn2d, 'rb') as handle:
    coordinates = pickle.load(handle)

#audio_files_paths = process_paths(audio_files)

# Count occurrences of each label
label_counts = Counter(labels)

# User input for label selection
label_prompt = "Available labels:\n"
for label, count in label_counts.items():
    label_prompt += f"  - Label {label}: {count} items\n"
label_prompt += "Enter a label to filter: "

selected_label = int(input(label_prompt))
if selected_label in label_counts:

    # Get the coordinates of the key sample for the selected label by averaging the coordinates of same label samples
    key_sample_coords = define_key_sample_coordinates(selected_label, coordinates)

    # Filter and plot    
    filter_and_plot(selected_label, audio_files, key_sample_coords, coordinates, labels)
else:
    print("Invalid label selected.")