import os
from playsound import playsound

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseEvent
import numpy as np
from pathlib import Path



def play_audio(file_path):
    """Play the selected audio file and wait until it finishes."""
    if os.path.exists(file_path):
        playsound(file_path)
    else:
        print(f"Error: File {file_path} not found.")


def process_paths(list_path_objects):
    """
    Process a list of path objects to extract subpaths starting from the 'Dropbox' folder
    and convert it to the appropriate OS format. Assuming the Dropbox is the the home directory.

    Args:
        list_path_objects (list): List of PosixPath objects.

    Returns:
        list: List of Path objects joined with the home directory.
    """
    # Get the user's home directory
    home_directory = Path.home()

    # Convert strings to Path objects and filter paths
    result_paths = []
    for current_path_object in list_path_objects:
        current_path_str = str(current_path_object)
        if 'Dropbox' in current_path_str:
            # Extract subpath starting from the 'Dropbox' folder
            subpath = current_path_str.split('Dropbox')[1]
            # Clean up the subpath
            subpath = subpath.replace('\\', '/').strip('/')
            result_path = home_directory.joinpath('Dropbox', subpath)

            result_paths.append(result_path)

    return result_paths


def calculate_distance(coord1, coord2):
    """Calculate Euclidean distance between two coordinates."""
    return np.linalg.norm(coord1 - coord2)


def filter_and_plot(selected_label, audio_files_paths, key_sample_coord, coordinates, labels):
    """Filter points by label and plot."""
    filtered_indices = [i for i, label in enumerate(labels) if label == selected_label]
    filtered_coordinates = coordinates[filtered_indices]
    filtered_files = [audio_files_paths[i] for i in filtered_indices]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot all filtered samples
    scatter = ax.scatter(filtered_coordinates[:, 0], filtered_coordinates[:, 1], c='blue', label=f'Label {selected_label}')

    # Plot the other labels with reduced opacity
    other_indices = [i for i, label in enumerate(labels) if label != selected_label]
    other_coordinates = coordinates[other_indices]
    ax.scatter(other_coordinates[:, 0], other_coordinates[:, 1], c='gray', alpha=0.2, label='Other Labels')


    # Highlight the key sample if it belongs to the selected label
    ax.scatter(key_sample_coord[0], key_sample_coord[1], c='red', s=100, marker='x', label='Key Sample')

    ax.set_title(f"Interactive Audio Plot (Label {selected_label})")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")

    # Callback function for clicks
    def on_click(event: MouseEvent):
        """Handle mouse click events."""
        if event.xdata is not None and event.ydata is not None:
            x_click, y_click = event.xdata, event.ydata
            distances = np.sqrt(np.sum((filtered_coordinates - np.array([x_click, y_click]))**2, axis=1))
            closest_index = np.argmin(distances)
            closest_distance = distances[closest_index]

            if closest_distance < 0.5:  # Threshold for valid selection
                selected_coord = filtered_coordinates[closest_index]
                selected_file = str(filtered_files[closest_index])

                # Calculate distance to key sample
                distance_to_key = calculate_distance(selected_coord, key_sample_coord)
                print(f"\n\tPlaying audio: {os.path.basename(selected_file)}")
                print(f"\tDistance to key sample: {distance_to_key:.2f}")

                # Update legend with distance
                legend_text = f"Distance to Key Sample: {distance_to_key:.2f}"
                ax.legend([scatter, ax.collections[-1]], [f'Label {selected_label}', legend_text])

                # Play audio
                play_audio(selected_file)
                fig.canvas.draw_idle()

    # Connect the click event
    fig.canvas.mpl_connect('button_press_event', on_click)

    ax.legend()
    plt.show()


def define_key_sample_coordinates(selected_label, coordinates):
    """Define the key sample coordinates for the selected label."""
    key_sample_indices = [i for i, label in enumerate(labels) if label == selected_label]
    key_sample_coords = np.mean(coordinates[key_sample_indices], axis=0)
    return key_sample_coords