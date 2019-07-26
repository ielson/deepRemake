import argparse
from imutils import paths
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from utilities.preprocessing import SimplePreprocessor
from utilities.datasets import SimpleDatasetLoader


import cv2

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True,
                help='Path to input dataset')
ap.add_argument('-n', '--neighbors', required=False, type=int, default=1,
                help='# of nearest neighbors for classification')
ap.add_argument('-j', '--jobs', required=False, type=int, default=-1,
                help='# of jobs for k-NN distance (-1 uses all available cores)')
args = vars(ap.parse_args())

# Get list of image paths
image_paths = list(paths.list_images(args['dataset']))

# Initialize SimplePreprocessor and SimpleDatasetLoader and load data and labels
print('[INFO]: Images loading....')
sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])
(data, labels) = sdl.load(image_paths, verbose=500)


# Reshape from (3000, 32, 32, 3) to (3000, 32*32*3=3072)
data = data.reshape((data.shape[0], 3072))

# Print information about memory consumption
print('[INFO]: Features Matrix: {:.1f}MB'.format(data.nbytes /(1024*1000.0)))

# Encode labels as integers
le = LabelEncoder()
# só tá mudando de gato, cachorro e panda para 0, 1 e 2
labels = le.fit_transform(labels)


# Split data into training (75%) and testing (25%) data
(train_x, test_x, train_y, test_y) = train_test_split(data, labels, test_size=0.25, random_state=42)

# Train and evaluate the k-NN classifier on the raw pixrandom_stateel intensities
print('[INFO]: Classification starting....')
model = KNeighborsClassifier(n_neighbors=args['neighbors'],
                             n_jobs=args['jobs'])

# o fit provavelmente deve separar os cluster para comparar depois
model.fit(train_x, train_y)
valoresPreditos = model.predict(test_x)
print(valoresPreditos)


print(classification_report(test_y, model.predict(test_x),
                            target_names=le.classes_))

classificadas = test_x.reshape(750, 32, 32, 3)
count = 0

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (200,128)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2

for imagem in classificadas:
    imagem = cv2.resize(imagem, (256, 256), cv2.INTER_LINEAR)
    cv2.putText(imagem, str(valoresPreditos[count]), bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
    cv2.imshow('classificadas', imagem)
    count+=1
    cv2.waitKey(0) & 0xff
