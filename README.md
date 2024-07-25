# Sign language Recognition Project

This project aims to capture and classify hand coordinates using a machine learning model. The system utilizes OpenCV for video capture and MediaPipe for hand tracking, alongside a K-Nearest Neighbors (KNN) classifier to recognize hand gestures.

## Features

- Capture hand coordinates using a webcam.
- Process and normalize the dataset.
- Train a KNN classifier to predict hand gestures.
- Display the predicted gesture in real-time on the video feed.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.7+
- OpenCV
- MediaPipe
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

You can install the required packages using pip:

```bash
pip install opencv-python mediapipe pandas numpy scikit-learn matplotlib
```

## Code Overview

### Importing Libraries

The following libraries are imported for the project:

```python
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
```

## Loading and Processing the Dataset
Load the dataset and split it into training and testing sets:

```python
dataset = pd.read_csv('hand_dataset_1000_24.csv')
X = dataset.iloc[:, 1:].values
Y = dataset.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)
```
## Normalize the dataset:

```python
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```
## Training the KNN Classifier
Train the KNN classifier and make predictions:

```python
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
```
## Evaluating the Model
Evaluate the model performance:

```python
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
```
## Error Analysis
Calculate and plot the error rate for different K values:

```python
error = []
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
```
## Real-Time Hand Gesture Recognition
Initialize MediaPipe and capture video from the webcam:

```python
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                coords = hand_landmarks.landmark
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                coords = list(np.array([[landmark.x, landmark.y] for landmark in coords]).flatten())
                coords = scaler.transform([coords])
                predicted = classifier.predict(coords)

                cv2.rectangle(image, (0,0), (100, 60), (245, 90, 16), -1)
                cv2.putText(image, 'CLASS', (20,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(predicted[0]), (20,45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows()
This project demonstrates how to capture and classify hand gestures using a machine learning model. The combination of OpenCV, MediaPipe, and Scikit-learn provides a robust framework for real-time hand gesture recognition.
```
