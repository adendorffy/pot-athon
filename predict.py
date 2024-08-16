import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, KBinsDiscretizer
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer, mean_squared_error
import pickle
import glob
from extract_features import *

def train_lr_model(df, alpha=10.0, degree=2):
    """
    Train and evaluate Linear Regression and Ridge Regression models.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing feature columns 'Area', 'Width', 'Height', and target column 'Bags'.
    degree (int): Degree of polynomial features to use for the models.
    """
    # Drop rows where 'Bags' column is NaN
    df = df.dropna(subset=['Bags'])
    
    # Extract feature columns and target variable
    X = df[['Area', 'Width', 'Height']]
    
    # Scaling the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Converting scaled features back to DataFrame
    X_scaled_df = pd.DataFrame(X_scaled, columns=['Area', 'Width', 'Height'])

    pca = PCA(n_components=3)  # Adjust the number of components as needed
    X_reduced = pca.fit_transform(X_scaled_df)

    y = df['Bags']

    binner = KBinsDiscretizer(n_bins=7, encode='ordinal', strategy='uniform', subsample=200_000)
    y_binned = binner.fit_transform(y.values.reshape(-1, 1)).ravel()
    
    # Define Mean Squared Error scorer
    mse_scorer = make_scorer(mean_squared_error)
    
    # Define KFold cross-validation procedure
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    # Create polynomial features
    poly_lr = PolynomialFeatures(degree=degree)
    X_poly_lr = poly_lr.fit_transform(X_reduced)
    
    # Initialize and evaluate Linear Regression model
    model_lr = LinearRegression()
    cv_scores = cross_val_score(model_lr, X_poly_lr, y=y_binned, cv=kf, scoring=mse_scorer)
    model_lr.fit(X_poly_lr, y)  # Train the model
    print(f'Mean Cross-Validation MSE LR: {cv_scores.mean()}')  # Print mean MSE
    
    # Create polynomial features for Ridge Regression
    poly_ridge = PolynomialFeatures(degree=degree)
    X_poly_ridge = poly_ridge.fit_transform(X_reduced)
    
    # Initialize and evaluate Ridge Regression model
    model_ridge = Ridge(alpha=alpha)  # Ridge regression with regularization
    cv_scores = cross_val_score(model_ridge, X_poly_ridge, y_binned, cv=kf, scoring=mse_scorer)
    model_ridge.fit(X_poly_ridge, y)  # Train the model
    print(f'Mean Cross-Validation MSE Ridge: {cv_scores.mean()}')  # Print mean MSE
    
    # Save trained models and polynomial feature transformers to disk
    with open('models/linear_regression_model.pkl', 'wb') as file: pickle.dump(model_lr, file)
    with open('models/ridge_regression_model.pkl', 'wb') as file: pickle.dump(model_ridge, file)
    with open('models/poly_lr_features.pkl', 'wb') as file: pickle.dump(poly_lr, file)
    with open('models/poly_ridge_features.pkl', 'wb') as file: pickle.dump(poly_ridge, file)

def evaluate_lr_model(df, degree=2):
    """
    Load trained models and feature transformers, make predictions, and save results.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing feature columns 'Area', 'Width', 'Height'.
    degree (int): Degree of polynomial features used in the models.
    """
    # Load trained models and feature transformers from disk
    with open('models/linear_regression_model.pkl', 'rb') as file: model_lr = pickle.load(file)
    with open('models/ridge_regression_model.pkl', 'rb') as file: model_ridge = pickle.load(file)
    with open('models/poly_lr_features.pkl', 'rb') as file: poly_lr = pickle.load(file)
    with open('models/poly_ridge_features.pkl', 'rb') as file: poly_ridge = pickle.load(file)
    
    # Extract feature columns from DataFrame
    X = df[['Area', 'Width', 'Height']]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Converting scaled features back to DataFrame
    X_scaled_df = pd.DataFrame(X_scaled, columns=['Area', 'Width', 'Height'])

    pca = PCA(n_components=3)  # Adjust the number of components as needed
    X_reduced = pca.fit_transform(X_scaled_df)
    
    # Transform features using the saved polynomial transformers
    X_poly_lr = poly_lr.transform(X_reduced)
    X_poly_ridge = poly_ridge.transform(X_reduced)
    
    # Make predictions using the loaded models
    y_lr = model_lr.predict(X_poly_lr)
    y_ridge = model_ridge.predict(X_poly_ridge)
    
    # Add predictions to the DataFrame
    df['Bags_lr'] = y_lr
    df['Bags_ridge'] = y_ridge
    
    # Save DataFrame with predictions to a CSV file
    df.to_csv("test_area_features.csv", index=False)

    return df

# Define file paths and directories
image_dir = "data_augmented/train_images/"
annotation_dir = "data_augmented/train_annotations"
labels_csv = "data/train_labels.csv"
features_csv = "train_area_features_augmented.csv"

# Uncomment the following lines for test setup
# image_dir = "data/test_images/test_images"
# annotation_dir = "data/test-annotations"
# labels_csv = "data/test_labels.csv"
# features_csv = "test_area_features.csv"

def generate_features(image_dir, annotation_dir, labels_csv, features_csv):
    """
    Generate feature data from images and annotations, and save it to a CSV file.
    
    Parameters:
    image_dir (str): Directory containing image files.
    annotation_dir (str): Directory containing annotation files.
    labels_csv (str): Path to the CSV file containing labels.
    features_csv (str): Path to save the generated feature CSV file.
    """
    # Load labels from CSV
    labels_df = pd.read_csv(labels_csv)
    data = []

    # Process each image file
    for image_file in glob.glob(image_dir + "/*.jpg"):
        result = process_image(image_file, annotation_dir, labels_df)   
        print(result)        
        data.append(result)
        
    # Create a DataFrame from the generated data
    df_test = pd.DataFrame(data)
    
    # Save the feature DataFrame to a CSV file
    df_test.to_csv(features_csv, index=False)

# Generate features for training or testing
generate_features(image_dir, annotation_dir, labels_csv, features_csv)

# Uncomment the following lines to train and evaluate models
train_lr_model(pd.read_csv("train_area_features.csv"),10, 2)
evaluate_lr_model(pd.read_csv("test_area_features.csv"), 2)
