from django.shortcuts import render
from django.http import Http404
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from skimage import feature
from imutils import build_montages
from imutils import paths
import numpy as np
import argparse
import cv2
import os
import uuid

# Create your views here.


def index(request):
    return render(request, 'detections/index.html')


def results(request):
    def quantify_image(image):
        # compute the histogram of oriented gradients feature vector for
        # the input image
        features = feature.hog(image, orientations=9, pixels_per_cell=(
            10, 10), cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")
        # return the feature vector
        return features

    def load_split(path):
        # grab the list of images in the input directory, then initialize
        # the list of data (i.e., images) and class labels
        imagePaths = list(paths.list_images(path))
        data = []
        labels = []

        # loop over the image paths
        for imagePath in imagePaths:
            # extract the class label from the filename
            label = imagePath.split(os.path.sep)[-2]

            # load the input image, convert it to grayscale, and resize
            # it to 200x200 pixels, ignoring aspect ratio
            image = cv2.imread(imagePath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (200, 200))

            # threshold the image such that the drawing appears as white
            # on a black background
            image = cv2.threshold(image, 0, 255,
                                  cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

            # quantify the image
            features = quantify_image(image)

            # update the data and labels lists, respectively
            data.append(features)
            labels.append(label)

        # return the data and labels
        return (np.array(data), np.array(labels))

    def load_test(path):
        # grab the list of images in the input directory, then initialize
        # the list of data (i.e., images) and class labels
        # imagePaths = list(paths.list_images(path))
        imagePaths = path
        data = []
        labels = []

        # loop over the image paths
        for imagePath in imagePaths:
            # extract the class label from the filename
            # label = "parkinson"
            label = "healthy"

            # load the input image, convert it to grayscale, and resize
            # it to 200x200 pixels, ignoring aspect ratio
            image = cv2.imread(imagePath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (200, 200))

            # threshold the image such that the drawing appears as white
            # on a black background
            image = cv2.threshold(image, 0, 255,
                                  cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

            # quantify the image
            features = quantify_image(image)

            # update the data and labels lists, respectively
            data.append(features)
            labels.append(label)

        # return the data and labels
        return (np.array(data), np.array(labels))

    if request.method == 'POST' and request.FILES.getlist('file'):
        files = request.FILES.getlist('file')
        picType = request.POST['pictureType']

        testingPath = []

        for file in files:
            fs = FileSystemStorage(
                location=settings.MEDIA_ROOT + '/uploads')
            filename = fs.save(uuid.uuid1().hex +
                               os.path.splitext(file.name)[1], file)
            uploaded_file_url = fs.url(filename)
            testing = os.path.sep.join(
                [settings.MEDIA_ROOT, 'uploads', filename])
            testingPath.append(testing)
        # define the path to the training and testing directories
        trainingPath = os.path.sep.join(
            [settings.MEDIA_ROOT, "datasets", picType])

        # loading the training and testing data
        (trainX, trainY) = load_split(trainingPath)
        (testX, testY) = load_test(testingPath)

        # encode the labels as integers
        le = LabelEncoder()
        trainY = le.fit_transform(trainY)
        testY = le.transform(testY)

        # initialize our trials dictionary
        trials = {}

        # loop over the number of trials to run
        for i in range(0, 100):
            # train the model
            model = RandomForestClassifier(n_estimators=100)
            model.fit(trainX, trainY)

            # make predictions on the testing data and initialize a dictionary
            # to store our computed metrics
            # predictions = model.predict(testX)
            # metrics = {}

            # compute the confusion matrix and and use it to derive the raw
            # accuracy, sensitivity, and specificity
            # cm = confusion_matrix(testY, predictions).flatten()

            # (tn, fp, fn, tp) = cm
            # np.seterr(invalid='ignore')
            # metrics["acc"] = (tp + tn) / float(cm.sum())
            # metrics["sensitivity"] = tp / float(tp + fn)
            # metrics["specificity"] = tn / float(tn + fp)

            # loop over the metrics
            # for (k, v) in metrics.items():
            #     # update the trials dictionary with the list of values for
            #     # the current metric
            #     l = trials.get(k, [])
            #     l.append(v)
            #     trials[k] = l

        # loop over our metrics
        # for metric in ("acc", "sensitivity", "specificity"):
        #     # grab the list of values for the current metric, then compute
        #     # the mean and standard deviation
        #     values = trials[metric]
        #     mean = np.mean(values)
        #     std = np.std(values)

        #     # show the computed metrics for the statistic
        #     # print(metric)
        #     # print("=" * len(metric))
        #     # print("u={:.4f}, o={:.4f}".format(mean, std))
        #     # print("")

        # randomly select a few images and then initialize the output images
        # for the montage
        idxs = np.arange(0, len(testingPath))
        # idxs = np.random.choice(idxs, size=(25,), replace=False)
        # images = []

        # The separation between health and Parkinson
        healthy = []
        parkinson = []

        # loop over the testing samples
        for i in idxs:
            # load the testing image, clone it, and resize it
            image = cv2.imread(testingPath[i])
            output = image.copy()
            output = cv2.resize(output, (128, 128))

            # pre-process the image in the same manner we did earlier
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (200, 200))
            image = cv2.threshold(image, 0, 255,
                                  cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

            # quantify the image and make predictions based on the extracted
            # features using the last trained Random Forest
            features = quantify_image(image)
            preds = model.predict([features])
            label = le.inverse_transform(preds)[0]

            basename = os.path.basename(testingPath[i])
            healthy.append(
                basename) if label == "healthy" else parkinson.append(basename)

            # draw the colored class label on the output image and add it to
            # the set of output images
            # color = (0, 255, 0) if label == "healthy" else (0, 0, 255)
            # image = cv2.putText(output, label, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            #                     color, 2)
            # cv2.imwrite(os.path.sep.join(
            #     [settings.MEDIA_ROOT, 'results', uuid.uuid4().hex + '.png', ]),  image, [
            #     int(cv2.IMWRITE_JPEG_QUALITY), 90])
            # images.append(output)
        context = {'healthy_list': healthy, 'parkinson_list': parkinson}
        return render(request, 'detections/result.html', context)


def datasets(request):
    healthy = []
    parkinson = []

    for file in os.listdir(os.path.join(settings.MEDIA_ROOT,  'datasets', 'spiral', 'healthy')):
        print(file)

    context = {'healthy_list': healthy, 'parkinson': parkinson}
    return render(request, 'detections/dataset.html')
