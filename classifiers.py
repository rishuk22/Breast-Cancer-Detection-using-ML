import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
import xgboost as xgb
import ast  # To safely evaluate strings

# Load the CSV file
df = pd.read_csv(r"E:\breastcancer\output_glcm_clahe.csv")

# Print the first few rows of the DataFrame to ensure it's loaded correctly
print(df.head())

# Print the distribution of classes in the 'Label' column
print("Class distribution:\n", df['Label'].value_counts())

# Extract features and labels from the DataFrame
X = []
y = []
for index, row in df.iterrows():
    label = row['Label']
    # Safely convert string representation of list to a Python list
    concatenated_vector = ast.literal_eval(row['GLCM Properties'])  
    X.append(concatenated_vector)
    y.append(label)

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define classifiers
classifiers = {
    "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42),
    "Support Vector Machine": SVC(class_weight='balanced', random_state=42),
    "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    "ANN": Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(len(np.unique(y)), activation='softmax')  # Number of classes
    ])
}

# Compile the ANN
classifiers["ANN"].compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train and evaluate each classifier
for name, classifier in classifiers.items():
    if name == "ANN":
        # Train the ANN classifier
        classifier.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
        # Make predictions
        y_pred = np.argmax(classifier.predict(X_test), axis=-1)
    else:
        # Train the classifier
        classifier.fit(X_train, y_train)
        # Make predictions
        y_pred = classifier.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Calculate TPR and TNR
    tn, fp, fn, tp = conf_matrix.ravel()  # Unpack confusion matrix
    tpr = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0  # True Positive Rate
    tnr = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0  # True Negative Rate

    # Print metrics
    print(f'\n{name} Classifier:')
    print(f'Accuracy: {accuracy:.2f} %')
    print(f'True Positives (TP): {tp}')
    print(f'True Negatives (TN): {tn}')
    print(f'False Positives (FP): {fp}')
    print(f'False Negatives (FN): {fn}')
    print(f'True Positive Rate: {tpr:.2f} %')
    print(f'True Negative Rate: {tnr:.2f} %')
    print('Confusion Matrix:\n', conf_matrix)

    print("---------------------------------------------------------------------------------")
