# -*- coding: utf-8 -*-

import os

from hskl.demo import dl_hyrank, load_hyrank
import hskl.classification as classification
import hskl.utils as utils

# Download, unpack, and load HyRANK dataset from current directory.
path = os.getcwd()
if not os.path.exists("HyRANK_satellite"):
    dl_hyrank(path)    
images, labels, _ = load_hyrank(path)

# Dimensional reduction using PCA, retain 99.9% image variance
pca = utils.pca_fit(images[0])
train, _ = utils.pca_apply(images[0], pca, 0.999)
test, _ = utils.pca_apply(images[1], pca, 0.999)
label = labels[0]
test_mask = labels[1]>0

# Train a classifier and predict test image labels
cl = classification.HyperspectralClassifier(
         method_name="LinearDiscriminantAnalysis")
cl.fit(train, label)
prediction = cl.predict(test)

# Visualization of training data, test prediction, and test ground truth
fig_objs_train = utils.overlay(train,label)
utils.save_overlay(fig_objs_train, "hyrank_train.png")

fig_objs_predict = utils.overlay(test,prediction*test_mask)
utils.save_overlay(fig_objs_predict, "hyrank_predict.png")

fig_objs_test = utils.overlay(test,labels[1])
utils.save_overlay(fig_objs_test, "hyrank_test.png")
