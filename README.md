# VGG_Classifier
 
#### Train to classify the BVH textures with labels

# How to use

### 1. Python environment
The Project is based on python 3.6x
Configure python environment according to requirements.txt

### 2. Datasets
Divided the Train and test data with a ratio of 9:1 and store them to motion_data/test and motion_data/train.

### 3. Train 

Train the datasets by running vgg_train.py(recommend) or running run.py for a user interface.

### 3.  Test the result
modify the model_path in line 70 in vgg_classify.py and run vgg_classify.py.