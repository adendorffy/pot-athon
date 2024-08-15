import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import make_scorer, mean_squared_error
import pickle
import glob
from extract_features import *

def train_lr_model(df, degree=2):
    df = df.dropna(subset=['Bags'])
    X = df[['Area', 'Width', 'Height']]
    y = df['Bags']

    mse_scorer = make_scorer(mean_squared_error)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    poly_lr = PolynomialFeatures(degree=1)
    X_poly_lr = poly_lr.fit_transform(X)
    model_lr = LinearRegression()
    cv_scores = cross_val_score(model_lr, X_poly_lr, y=y, cv=kf, scoring=mse_scorer)
    model_lr.fit(X_poly_lr, y)
    print(f'Mean Cross-Validation MSE LR: {cv_scores.mean()}')

    poly_ridge = PolynomialFeatures(degree=1)
    X_poly_ridge = poly_ridge.fit_transform(X)
    model_ridge = Ridge(alpha=50.0)
    cv_scores = cross_val_score(model_ridge, X_poly_ridge, y, cv=kf, scoring=mse_scorer)
    model_ridge.fit(X_poly_ridge, y)
    print(f'Mean Cross-Validation MSE Ridge: {cv_scores.mean()}')

    with open('models/linear_regression_model.pkl', 'wb') as file: pickle.dump(model_lr, file)
    with open('models/ridge_regression_model.pkl', 'wb') as file: pickle.dump(model_ridge, file)
    with open('models/poly_lr_features.pkl', 'wb') as file: pickle.dump(poly_lr, file)
    with open('models/poly_ridge_features.pkl', 'wb') as file: pickle.dump(poly_ridge, file)

def evaluate_lr_model(df, degree=2):
    with open('models/linear_regression_model.pkl', 'rb') as file: model_lr = pickle.load(file)
    with open('models/ridge_regression_model.pkl', 'rb') as file: model_ridge = pickle.load(file)
    with open('models/poly_lr_features.pkl', 'rb') as file: poly_lr = pickle.load(file)
    with open('models/poly_ridge_features.pkl', 'rb') as file: poly_ridge = pickle.load(file)
        
    X = df[['Area', 'Width', 'Height']]
    X_poly_lr = poly_lr.transform(X)
    X_poly_ridge = poly_ridge.transform(X)
    
    y_lr = model_lr.predict(X_poly_lr)
    y_ridge = model_ridge.predict(X_poly_ridge)
    df['Bags_lr'] = y_lr
    df['Bags_ridge'] = y_ridge
    df.to_csv("test_area_features.csv", index=False)

    return df

# # # Train
# image_dir = "data/train_images/train_images"
# annotation_dir = "data/train-annotations"
# labels_csv = "data/train_labels.csv"
# features_csv = "train_area_features.csv"

# Test
image_dir = "data/test_images/test_images"
annotation_dir = "data/test-annotations"
labels_csv = "data/test_labels.csv"
features_csv = "test_area_features.csv"

def generate_features(image_dir, annotation_dir, labels_csv, features_csv):
    labels_df = pd.read_csv(labels_csv)
    data = []

    for image_file in glob.glob(image_dir+"/*.jpg"):
        print(image_file)
        result = process_image(image_file, annotation_dir, labels_df)           
        data.append(result)
            
    df_test = pd.DataFrame(data)
    df_test.to_csv(features_csv, index=False)

# generate_features(image_dir, annotation_dir, labels_csv, features_csv)
train_lr_model(pd.read_csv("train_area_features.csv"), 2)
evaluate_lr_model(pd.read_csv("test_area_features.csv"), 2)