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
    stick_area_mm2 = stick_length_mm * stick_width_mm
    # Process each directory
    for directory in directories:
        # Load the _annotations.csv file
        annotations_file = os.path.join(directory, '_annotations.csv')
        df = pd.read_csv(annotations_file)
        
        # Extract the pothole ID
        df['pothole_id'] = df['filename'].str.extract(r'p(\d+)_')[0]
        
        # Calculate the area for each class
        df['area'] = (df['xmax'] - df['xmin']) * (df['ymax'] - df['ymin'])
        
        # Pivot the data to get separate columns for pothole area and stick area
        pivot_df = df.pivot_table(index='pothole_id', columns='class', values='area', aggfunc='first').reset_index()
        
        # Rename the columns for clarity
        pivot_df = pivot_df.rename(columns={'potholes': 'pothole_area', 'L': 'stick_area'})
        
        # Save the result to a new CSV file
        output_file = os.path.join(directory, f'{os.path.basename(directory)}.csv')

        print(f"Processed {annotations_file}")
        
        # Calculate the area of the stick in pixels (already computed in previous steps)
        pivot_df['stick_area'] = pivot_df['stick_area'].fillna(1)  # To avoid division by zero or NaN
        
        # Calculate the conversion factor from pixels to mm²
        pivot_df['conversion_factor'] = stick_area_mm2 / pivot_df['stick_area']
        
        # Convert the pothole area to square millimeters
        pivot_df['pothole_area_mm2'] = pivot_df['pothole_area'] * pivot_df['conversion_factor']
        
        final_df = pivot_df.drop(['stick_area','pothole_area','conversion_factor'], axis=1)

        # Save the updated DataFrame back to the CSV file
        final_df.to_csv(output_file, index=False)

        print(f"Processed {output_file} and converted pothole areas to square millimeters.")

import pandas as pd
import os

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
