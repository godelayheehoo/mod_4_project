# An X-ray is worth a thousand words

This project was completed for Flatiron's Data Science program by James Shaw, Michael Wang, and Bobby Williams.

The data used for this project can be found on Kaggle at this link: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

The dataset contains jpeg images of pediatric x-rays for patients with and without pneumonia. The goal is to reduce the time it takes to diagnose pneumonia in a patient by taking in an image of an x-ray and classifying the image as "normal" or "pneumonia".

A Convolutional Neural Network was trained to process the images and provide a classification. Recall was the primary metric used to determine model performance and by iterating through networks with different layers and configurations we were able to maximize the Recall at 97% for the test image set.

This repo contains the following files:

- EDA.ipynb - the notebook used for exploratory data analysis with visualizations of the data distribution.
- README.md - this document provides a brief overview of the project and the files contained in the repo.
- x-ray.pdf - the presentation slide deck.
- executive_notebook.ipynb - the main notebook discussing the approach to the problem, the model, and performance evaluation.
- exportable_vis_work.ipynb - this notebook explains the visualizations used throughout the project.
- models - this directory contains each of the saved models found in the models.ipynb notebook.
- models.ipynb - this notebook contains each of the models that were iterated through when attempting to maximize Recall.
- project_4_exportable_env.yml - this file contains the environment used for the project.
- other miscellaneous media files used for the project.
