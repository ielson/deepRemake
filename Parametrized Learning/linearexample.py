import sys
import cv2
import numpy as np


# Initialize class labels and set the seed of our pseudo-random number generator
# '1' is chosen as the seed because it gives the 'correct classification'
labels = ['dog', 'cat', 'panda']
np.random.seed(1)

# Randomly initialize the weight and bias vectors between 0 and 1
w = np.random.randn(3, 3072)
b = np.random.randn(3)

try:
    # Load image, resize it (ignoring the aspect ratio) and flatten it
    original = cv2.imread('beagle.png')
    # Remember that in the previous example knn in order to flatten the image we used: data.reshape((data.shape[0], 3072))
    image = cv2.resize(original, (32, 32)).flatten()
except(cv2.error):
    print("Tthe image is either bad or doesn't exist.")
    sys.exit()
    

# Compute the output scores
scores = w.dot(image) + b

# Loop over the scores and labels to display them
for label, score in zip(labels, scores):
    print('[INFO]: {}: {:.2f}'.format(label, score))

# Draw the label with the highest score on the image as our prediction
cv2.putText(original, 'Label: {}'.format(labels[np.argmax(scores)]), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Display our input image
cv2.imshow("Image", original)
cv2.waitKey(0)
