from ctypes import sizeof
import cv2
import numpy as np
from skimage.feature.texture import graycomatrix, graycoprops
from skimage.feature import local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.feature import hog

def rgb_histogram(image, bins=256):
    hist_features = []
    for i in range(3):  # RGB Channels
        hist, _ = np.histogram(image[:, :, i], bins=bins, range=(0, 256), density=True)
        hist_features.append(hist)
    return np.concatenate(hist_features)

def hu_moments(image):
    # Convert to grayscale if the image is in RGB format
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    moments = cv2.moments(gray)
    hu_moments = cv2.HuMoments(moments).flatten()
    return hu_moments

def filter_image(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_color = np.array([20, 0, 0])    # gray/silver
    upper_color = np.array([270, 40, 100])    
    mask = cv2.inRange(hsv, lower_color, upper_color)

    #lower_orange = np.array([0, 50, 80])
    #upper_orange = np.array([25, 255, 255])
    #mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
    #mask = cv2.bitwise_and(mask, cv2.bitwise_not(mask_orange))
    image = cv2.bitwise_and(image, image, mask=mask)
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    hist_features = []
    for i in range(3):  # RGB Channels
        hist, _ = np.histogram(image[:, :, i], bins=256, range=(1, 256), density=True)
        hist_features.append(hist)
    return np.concatenate(hist_features)


def hog_features(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Resize to standard size for consistency
    gray_resized = cv2.resize(gray, (64, 64))
    
    hog_features_vector = hog(
        gray_resized,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm='L2-Hys',
        feature_vector=True
    )
    
    return hog_features_vector



def luv_histogram(image, bins=32):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
    hist_features = []
    for i in range(3):  # RGB Channels
        hist, _ = np.histogram(image[:, :, i], bins=bins, range=(0, 256), density=True)
        hist_features.append(hist)
    return np.concatenate(hist_features)

def glcm_features(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    glcm = graycomatrix(gray, distances=distances, angles=angles, levels=levels, symmetric=symmetric, normed=normed)
    contrast = graycoprops(glcm, 'contrast').flatten()
    dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
    homogeneity = graycoprops(glcm, 'homogeneity').flatten()
    energy = graycoprops(glcm, 'energy').flatten()
    correlation = graycoprops(glcm, 'correlation').flatten()
    asm = graycoprops(glcm, 'ASM').flatten()
    return np.concatenate([contrast, dissimilarity, homogeneity, energy, correlation, asm])

def local_binary_pattern_features(image, P=8, R=1):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray, P, R, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2), density=True)
    return hist

def extract_features_from_image(image):
    
    # 1. RGB Histogram
    reduced_hist=filter_image(image)
    luv_hist = luv_histogram(image)
    #print(np.shape(luv_hist))
    hog =hog_features(image)
    #print(np.shape(hog))

    # 2. Hu Moments
    hu_features = hu_moments(image)
    #print(np.shape(hu_features))

    # 3. GLCM Features
    glcm_features_vector = glcm_features(image)
    #print(np.shape(glcm_features_vector))
   
    # 4. Local Binary Pattern (LBP)
    lbp_features = local_binary_pattern_features(image)
    #print(np.shape(lbp_features))
    hist_features = rgb_histogram(image)
    #### Add more feature extraction methods here ####
    
    
    
    
    ##################################################
    
    
    # Concatenate all feature vectors
    image_features = np.concatenate([ hu_features,reduced_hist,hist_features, glcm_features_vector, lbp_features, hog,luv_hist])
    
    return image_features


def perform_pca(data, num_components):
    """
    Perform Principal Component Analysis (PCA) on the input data.

    Parameters:
    - data (numpy.ndarray): The input data with shape (n_samples, n_features).
    - num_components (int): The number of principal components to retain.

    Returns:
    - data_reduced (numpy.ndarray): The data transformed into the reduced PCA space.
    - top_k_eigenvectors (numpy.ndarray): The top k eigenvectors.
    - sorted_eigenvalues (numpy.ndarray): The sorted eigenvalues.
    """
    # Step 1: Standardize the Data
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    std_dev[std_dev==0]=1
    print(sum(std_dev==0))
    data_standardized = (data - mean) / std_dev
    data_standardized = np.nan_to_num(data_standardized, nan=0.0, posinf=0.0, neginf=0.0)

    # Step 2: Compute the Covariance Matrix
    covariance_matrix = np.cov(data_standardized, rowvar=False)


    # Step 3: Calculate Eigenvalues and Eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Step 4: Sort Eigenvalues and Eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Step 5: Select the top k Eigenvectors
    top_k_eigenvectors = sorted_eigenvectors[:, :num_components]

    # Step 6: Transform the Data using the top k eigenvectors
    data_reduced = np.dot(data_standardized, top_k_eigenvectors)

    # Return the real part of the data (in case of numerical imprecision)
    data_reduced = np.real(data_reduced)

    return data_reduced



def train_svm_model(features, labels, test_size=0.2):
    """
    Trains an SVM model and returns the trained model.

    Parameters:
    - features: Feature matrix of shape (B, F)
    - labels: Label matrix of shape (B, C) if one-hot encoded, or (B,) for single labels
    - test_size: Proportion of the data to use for testing (default is 0.2)

    Returns:
    - svm_model: Trained SVM model
    """
    # Check if labels are one-hot encoded, convert if needed
    if labels.ndim > 1 and labels.shape[1] > 1:
        labels = np.argmax(labels, axis=1)  # Convert one-hot to single label per sample

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=42)

    # Create an SVM classifier (you can modify kernel or C as needed)
    svm_model = SVC(kernel='rbf', C=1.0)

    # Train the model
    svm_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = svm_model.predict(X_test)

    # Evaluate and print accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Test Accuracy: {accuracy:.2f}')
    y_pred = svm_model.predict(X_train)

    # Evaluate and print accuracy
    accuracy = accuracy_score(y_train, y_train)
    print(f'Train Accuracy: {accuracy:.2f}')
    # Return the trained model
    return svm_model