import os
import pandas as pd

def remove_test_images():
    # Path to your CSV file
    csv_file = 'data-v6/test/_annotations.csv'

    # Path to the directory containing the images to exclude
    exclude_dir = 'data-v6/test/their_test'

    # Path to save the excluded rows
    excluded_csv = 'data-v6/test/their_test/_annotations.csv'

    # Get a list of image filenames (without path) in the exclude directory
    exclude_images = {os.path.basename(file) for file in os.listdir(exclude_dir)}

    # Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Split the DataFrame into rows to keep and rows to remove
    df_to_remove = df[df['filename'].isin(exclude_images)]
    filtered_df = df[~df['filename'].isin(exclude_images)]

    # Save the filtered DataFrame back to the CSV file
    filtered_df.to_csv(csv_file, index=False)

    # Save the removed rows into a new CSV file
    df_to_remove.to_csv(excluded_csv, index=False)

    print(f"Rows associated with images in {exclude_dir} have been removed.")

# Function to calculate the area of an element
def calculate_area(row):
    return (row['xmax'] - row['xmin']) * (row['ymax'] - row['ymin'])


def create_csv_per_set(directories):
    # Known dimensions of the stick in millimeters
    stick_length_mm = 500
    stick_width_mm = 10

    # Stick area in square millimeters
    # Process each directory
    for directory in directories:
        # Load the _annotations.csv file
        annotations_file = os.path.join(directory, '_annotations.csv')
        # Load the annotations
        annotations = pd.read_csv(annotations_file)

        # Create an empty list to store results
        results = []

        for _, row in annotations.iterrows():
            filename = row['filename']
            xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
            
            # Extract pothole_number from filename
            pothole_number = filename.split('_')[0][1:]
            
            # Calculate width and height of the bounding box
            width = xmax - xmin
            height = ymax - ymin
            
            # Calculate aspect ratio
            aspect_ratio = width / height
            
            # Calculate the area
            area = width * height
            
            # Append the result to the list
            results.append({
                'pothole_id': pothole_number,
                'aspect_ratio': aspect_ratio,
                'pothole_area_mm2': area
            })
            
         # Save the result to a new CSV file
        output_file = os.path.join(directory, f'{os.path.basename(directory)}.csv')

        # Convert results to a DataFrame
        results_df = pd.DataFrame(results)
        print(results_df)
        results_df.to_csv(output_file, index=False)
        print(f"Processed {output_file} and converted pothole areas to square millimeters.")


def add_labels(directories):
    combined_labels_df = pd.read_csv('combined_train_labels.csv')

    # Process each directory
    for directory in directories:
        # Load the corresponding CSV file
        csv_file = os.path.join(directory, f'{os.path.basename(directory)}.csv')
        df = pd.read_csv(csv_file)

        # Merge the two datasets on 'pothole_id'
        merged_df = pd.merge(df, combined_labels_df, how='left', on='pothole_id')

        # Save the merged dataset back to a CSV file (if needed)
        merged_df.to_csv(csv_file, index=False)

        print(f"Processed {csv_file} and added bag usage information.")


if __name__ == "__main__":
    # run this to remove their test images data
    remove_test_images()
    # Directories to process
    directories = ['data-v6/test', 'data-v6/train', 'data-v6/valid']
    create_csv_per_set(directories)
    add_labels(directories)
    # Load the datasets
    train_df = pd.read_csv('data-v6/train/train.csv')
    valid_df = pd.read_csv('data-v6/valid/valid.csv')
    test_df = pd.read_csv('data-v6/test/test.csv')

    # Combine train, validation, and test datasets
    full_df = pd.concat([train_df, valid_df, test_df])
    full_df.to_csv('all_potholes.csv', index=False)

