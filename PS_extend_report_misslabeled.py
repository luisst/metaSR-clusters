import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# --- Paths ---
input_path = Path(r"C:\Users\luis2\Dropbox\clustering\meta-SR\saved_model\noisy_all_18K_f3-s3q2-aug0.8_c15_500-s2q2\noisy_all_18K_f3-s3q2-aug0.8_c15_500-s2q2_results_misslabeled.txt")
output_folder = input_path.parent / "plots"
output_folder.mkdir(parents=True, exist_ok=True)

# --- Load Data ---
df = pd.read_csv(input_path, sep='\t')

# Extract speaker ID from filename
df['speaker_id'] = df['filename'].apply(lambda x: x.split('_')[1])

# Compute total predictions and accuracy (%)
df['total'] = df['misslabeled_count'] + df['correct_count']
df['accuracy'] = 100 * df['correct_count'] / df['total']

# --- Speaker-level stats ---
speaker_stats = df.groupby('speaker_id').agg(
    total_files=('filename', 'count'),
    total_correct=('correct_count', 'sum'),
    total_mislabeled=('misslabeled_count', 'sum'),
    total_preds=('total', 'sum')
)
speaker_stats['overall_accuracy'] = 100 * speaker_stats['total_correct'] / speaker_stats['total_preds']
speaker_stats = speaker_stats.sort_values('overall_accuracy', ascending=False)

# Round values
speaker_stats_rounded = speaker_stats.copy()
speaker_stats_rounded['overall_accuracy'] = speaker_stats_rounded['overall_accuracy'].round(1)
speaker_stats_rounded[['total_correct', 'total_mislabeled', 'total_preds']] = speaker_stats_rounded[['total_correct', 'total_mislabeled', 'total_preds']].round(2)

# --- Save speaker stats ---
speaker_stats_path = output_folder / "speaker_stats_summary.tsv"
speaker_stats_rounded.to_csv(speaker_stats_path, sep='\t')

# --- Plot 1: Bar plot of overall accuracy ---


plt.figure(figsize=(10, 6))

# Add the number of filesper speaker to the speaker name in X-axis. Trim name to first 6 characters for better readability
speaker_labels = []
for speaker, row in speaker_stats_rounded.iterrows():
    if len(speaker) > 6:
        # If speaker ID is longer than 5 characters, trim it
        trimmed_speaker = speaker[:5]  # Trim speaker ID to first 6 characters
    else:
        trimmed_speaker = speaker

    speaker_labels.append(f"{trimmed_speaker}({int(row['total_files'])})")

speaker_stats_rounded.index = speaker_labels

sns.barplot(x=speaker_stats_rounded.index, y=speaker_stats_rounded['overall_accuracy'], palette='viridis')
plt.ylabel("Overall Accuracy (%)")
plt.xlabel("Speaker ID")
plt.title("Speaker Recognition Accuracy per Speaker")
plt.xticks(rotation=45)
plt.tight_layout()
plt.axhline(y=50, color='r', linestyle='--', label='50% Threshold')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(output_folder / "accuracy_per_speaker_barplot.png")
plt.close()

# --- Outlier detection (accuracy < 2%) ---
low_accuracy_threshold = 2.0
outliers = df[df['accuracy'] < low_accuracy_threshold]
outliers = outliers[['filename', 'accuracy']].sort_values('accuracy')
outliers['accuracy'] = outliers['accuracy'].round(1)

# Save outliers
outliers_path = output_folder / "low_accuracy_outliers.tsv"
outliers.to_csv(outliers_path, sep='\t', index=False)

# Save full file-level info (rounded)
df_export = df.copy()
df_export['accuracy'] = df_export['accuracy'].round(1)
df_export[['correct_count', 'misslabeled_count', 'total']] = df_export[['correct_count', 'misslabeled_count', 'total']].round(2)

# Sort by accuracy
df_export = df_export.sort_values('accuracy', ascending=False)

df_export_path = output_folder / "full_accuracy_per_file.tsv"
df_export.to_csv(df_export_path, sep='\t', index=False)
