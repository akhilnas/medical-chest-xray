# Deep Learning for Medical Applications

This end-to-end project deals with the issue of incorrect diagonisis of X-ray images in the cases of Pneumonia. This is intended to be used as a tool by radiologists to assist them in their jobs.

The project is deployed using AWS Elastic Beanstalk and is available on the <a href= http://medicalchestxray-env-1.eba-gctyw3df.us-east-2.elasticbeanstalk.com/ >link</a> (Note: Due to AWS Charges and Budget constraints the link may not be active at this moment).

The data that is being used in this project is available on <a href = https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia>Kaggle</a>.

In case the Webpage is not available this repository can be pulled and the webpage can be hosted on your local machine with the following steps.

## Step 1: Download Repository and Data
```
$git pull https://github.com/akhilnas/medical-chest-xray.git
```
Download the data using the Kaggle link.

## Step 2: Install Dependencies
```
$pip install -r requirements.txt
```

## Step 3: Run Jupyter Notebook
This Notebook will generate the necessary dictionary of model parameters that will be used in the downstream processes. This is the training phase of the process wherein the model is trained on the data.

## Step 4: Run Flask Application
```
$python application.py
```
Click on the Link mentioned in the Terminal Window and the webpage should be displayed.



Proposed Extension: To acquire data that has the locaclity i.e a bounding box for the Pneumonia affliction in the X-ray images.

