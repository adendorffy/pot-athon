import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import make_scorer, mean_squared_error
import pickle
import glob
from extract_features import *

def train_lr_model(df, degree=2):

    df = df.dropna(subset=['Bags'])
    X = df[['Area', 'Width', 'Height']]
    y = df['Bags']
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)

    model_lr = LinearRegression()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    mse_scorer = make_scorer(mean_squared_error)
    cv_scores = cross_val_score(model_lr, X_poly, y=y, cv=kf, scoring=mse_scorer)
    print(f'Mean Cross-Validation MSE: {cv_scores.mean()}')
    
    model_lr.fit(X_poly, y)

    with open('models/linear_regression_model.pkl', 'wb') as file:
        pickle.dump(model_lr, file)
        
    with open('models/poly_features.pkl', 'wb') as file:
        pickle.dump(poly, file)

def evaluate_lr_model(df, degree=2):
    with open('models/linear_regression_model.pkl', 'rb') as file:
        model_lr = pickle.load(file)
        
    with open('models/poly_features.pkl', 'rb') as file:
        poly = pickle.load(file)
        
    X = df[['Area', 'Width', 'Height']]
    X_poly = poly.transform(X)
    
    y = model_lr.predict(X_poly)
    df['Bags'] = y
    df.to_csv("test_area_with_bags.csv", index=False)

    return df

# # Train
# image_dir = "data/train_images/train_images"
# annotation_dir = "data/train-annotations"
# labels_csv = "data/train_labels.csv"
# features_csv = "train_area_features.csv"

# # Test
image_dir = "data/test_images/test_images"
annotation_dir = "data/test-annotations"
labels_csv = "data/test_labels.csv"
features_csv = "test_area_features.csv"

def generate_features(image_dir, annotation_dir, labels_csv, features_csv):
    labels_df = pd.read_csv(labels_csv)
    data = []

    for image_file in glob.glob(image_dir+"/*.jpg"):
        result = process_image(image_file, annotation_dir, labels_df)           
        data.append(result)
            
    df_test = pd.DataFrame(data)
    df_test.to_csv(features_csv, index=False)

generate_features(image_dir, annotation_dir, labels_csv, features_csv)
train_lr_model(pd.read_csv("train_area_features.csv"), 2)
evaluate_lr_model(pd.read_csv("test_area_features.csv"), 2)