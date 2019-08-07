# Visualize Convnets DDD

Visualize the activations of network trained for driver distraction detection using StateFarm database 



######################### INFORMATION #################################


This repository consists of CNN visualization of intermediate layers of a driver distraction detection using StateFarm database. Two models were trained with similar structure and are visualised here. One that classifies only mobile based distraction and another that classifies all distractions.

Trained model and corresponding files are in : https://github.com/n-akram/DriverDistractionDetection

Classes used by model classifying only mobile based distractions: C0 to C4

Implementation from: https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/5.4-visualizing-what-convnets-learn.ipynb


Database used: StateFarm driver distraction detection.

https://www.kaggle.com/c/state-farm-distracted-driver-detection/data


#################### Technical INFORMATION ##############################

Implemented using: forVis environment

Activate using : source forVis/bin/activate


Environment details:
Python version: 3.5.2
Tensorflow backend : 1.14.0 (with GPU)
Keras : 2.2.4
Open CV : 4.1.0

System configuration:
OS: Linux
CPU: Intel core i9-9900K
GPU: GeForce RTX 2080 Ti

############################ To Run #####################################

1. Create virtual environment using: 
    virtualenv forVis
    Virtualenv packge can be installed using : pip install virtualenv
    
2. Install the relevant packages from "requirements.txt"
    pip install -r requirements.txt

3. Activate the environment: 
    source forVis/bin/activate

4. Run "python main.py" : for visualisation of model trained for all classes

5. Run "python main.py -m": for visualisation of model trained for mobile based distraction classes

6. Note: Change the path to test image to visualise convNets given input from test set
