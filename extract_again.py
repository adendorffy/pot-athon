import pandas as pd
import os 

directories = ['data-v6/test']

for directory in directories:
    # Load the annotation file
    annotations = pd.read_csv(f'{directory}/_annotations.csv')
    numbers_to_check = [
        103, 104, 105, 108, 114, 143, 144, 406, 434, 450, 470, 473, 479,
        1040, 1086, 1115, 1134, 1161, 1162, 1181, 1198, 1205, 1250, 1270,
        1278, 1280, 1296, 1409, 1430, 1438
    ]

    # Convert numbers to strings
    numbers_to_check_str = [str(num) for num in numbers_to_check]
    
    # Function to calculate area and aspect ratio
    def calculate_pothole_metrics(annotations):
        pothole_metrics = []
        test_metrics = []
        for filename in annotations['filename'].unique():
            number = filename.split('_')[0][1:]

            file_annotations = annotations[annotations['filename'] == filename]

            # Check if there is a stick object ('L') in the file's annotations
            if 'L' in file_annotations['class'].values:
                stick_row = file_annotations[file_annotations['class'] == 'L'].iloc[0]
                stick_length_pixels = stick_row['xmax'] - stick_row['xmin']
                stick_width_pixels = stick_row['ymax'] - stick_row['ymin']

                # Calculate conversion factors
                length_conversion_factor = 500 / stick_length_pixels
                width_conversion_factor = 4 / stick_width_pixels

                # Extract pothole dimensions and calculate area and aspect ratio
                pothole_rows = file_annotations[file_annotations['class'] == 'potholes']
                for _, row in pothole_rows.iterrows():
                    pothole_length_pixels = row['xmax'] - row['xmin']
                    pothole_width_pixels = row['ymax'] - row['ymin']
                    pothole_area_pixels = pothole_length_pixels * pothole_width_pixels

                    pothole_area_mm2 = pothole_area_pixels * length_conversion_factor * width_conversion_factor
                    aspect_ratio = pothole_length_pixels / pothole_width_pixels

                    if number in numbers_to_check_str:
                        test_metrics.append({
                        'filename': number,
                        'pothole_area_mm2': pothole_area_mm2,
                        'aspect_ratio': aspect_ratio
                    })
                    pothole_metrics.append({
                        'filename': number,
                        'pothole_area_mm2': pothole_area_mm2,
                        'aspect_ratio': aspect_ratio
                    })
            else:
                # Handle the case where no stick ('L') is found
                print(f"No stick ('L') found in {filename}. Skipping...")

        return pd.DataFrame(pothole_metrics), pd.DataFrame(test_metrics)

    # Calculate the metrics for all images
    pothole_metrics, test_metrics = calculate_pothole_metrics(annotations)

    # Save or display the results
    print(pothole_metrics)
    print(test_metrics)
    pothole_metrics.to_csv(f'{directory}/pothole_metrics.csv', index=False)
    test_metrics.to_csv(f'test_potholes.csv', index=False)


def add_labels(directories):
    combined_labels_df = pd.read_csv('combined_train_labels.csv')

    # Process each directory
    for directory in directories:
        # Load the corresponding CSV file
        csv_file = os.path.join(directory, f'pothole_metrics.csv')
        df = pd.read_csv(csv_file)
        df['pothole_id'] = df['filename']
        df = df.drop(['filename'], axis=1)
        # Merge the two datasets on 'pothole_id'
        merged_df = pd.merge(df, combined_labels_df, how='left', on='pothole_id')
    
        # Save the merged dataset back to a CSV file (if needed)
        merged_df.to_csv(csv_file, index=False)

        print(f"Processed {csv_file} and added bag usage information.")

# add_labels(directories)

# Load the datasets
train_df = pd.read_csv('data-v6/train/train1.csv')
valid_df = pd.read_csv('data-v6/valid/valid1.csv')
test_df = pd.read_csv('data-v6/test/test1.csv')

# Combine train, validation, and test datasets
full_df = pd.concat([train_df, valid_df, test_df])
full_df['Pothole number'] = full_df['pothole_id']
full_df = full_df.drop(['pothole_id'], axis=1)

full_df.to_csv('all_potholes.csv', index=False)
