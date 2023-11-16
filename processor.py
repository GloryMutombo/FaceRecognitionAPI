import os
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


def predict_sim(new_img_path):
    # Define your dataset path
    dataset_path = 'uploads/students/'

    # Step 1: Load and preprocess the images
    images, labels = load_images(dataset_path)

    # Step 2: Compute the mean face
    mean_face = compute_mean_face(images)

    # Step 3: Compute Eigenfaces using PCA
    eigenfaces = compute_eigenfaces(images)

    # Step 4: Project images onto the Eigenfaces
    eigenface_projections = project_onto_eigenfaces(images, mean_face, eigenfaces)

    # Step 5: Train a Model
    # accuracy = train_model(eigenface_projections, labels)
    accuracy, tuned_knn_model = train_model_with_tuning(eigenface_projections, labels)

    # Step 6: Load and preprocess the new image
    new_image = preprocess_new_image(new_img_path)

    # Step 7: Project the new image onto the Eigenfaces
    new_image_features = project_new_image(new_image, mean_face, eigenfaces)

    # Step 8: Compare against the dataset images
    distances = compare_images(new_image_features, eigenface_projections)

    # Step 9: Make a prediction
    prediction = make_prediction(distances, labels)

    data = {'accuracy': round(accuracy * 100), 'prediction': prediction}

    return data


# Step 1: Load and preprocess the images
def load_images(dataset_path):
    images = []
    labels = []
    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        for filename in os.listdir(person_dir):
            img_path = os.path.join(person_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read images in grayscale
            img = cv2.resize(img, (64, 128))  # Resize if needed
            images.append(img.flatten())
            labels.append(person_name)
    return np.array(images), np.array(labels)


# Step 2: Compute the mean face
def compute_mean_face(images):
    return np.mean(images, axis=0)


# Step 3: Compute Eigenfaces using PCA
def compute_eigenfaces(images, num_components=200):
    pca = PCA(n_components=num_components, whiten=True)
    pca.fit(images)
    eigenfaces = pca.components_
    return eigenfaces


# Step 4: Project images onto the Eigenfaces
def project_onto_eigenfaces(images, mean_face, eigenfaces):
    images_centered = images - mean_face
    eigenface_projections = np.dot(images_centered, eigenfaces.T)
    return eigenface_projections


# Step 5: Train a Model
def train_model(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


# Step 6:Load and preprocess the new image
def preprocess_new_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 128))
    return img.flatten()


# Step 7: Project the new image onto the Eigenfaces
def project_new_image(image, mean_face, eigenfaces):
    image_centered = image - mean_face
    eigenface_projection = np.dot(image_centered, eigenfaces.T)
    return eigenface_projection


# Step 8: Compare against the dataset images
def compare_images(new_image_features, dataset_features):
    # Use Euclidean distance as a similarity measure
    distances = np.linalg.norm(dataset_features - new_image_features, axis=1)
    return distances


# Step 9: Make a prediction
def make_prediction(distances, labels, threshold=1000):
    min_distance_index = np.argmin(distances)
    min_distance = distances[min_distance_index]
    predicted_label = labels[min_distance_index]

    if min_distance < threshold:
        return predicted_label
    else:
        return "Unknown"


# Testing other strategies
def train_model_with_tuning(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Define the parameter grid to search
    param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}  # Experiment with different values

    # Initialize KNN classifier
    knn = KNeighborsClassifier()

    # Use GridSearchCV for parameter tuning
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Get the best parameters from the grid search
    best_params = grid_search.best_params_
    print(f"Best Parameters: {best_params}")

    # Train the model with the best parameters
    best_knn_model = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'])
    best_knn_model.fit(X_train, y_train)

    # Evaluate the model on the test set
    y_pred = best_knn_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, best_knn_model
