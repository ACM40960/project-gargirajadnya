[![python](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org) ![Static Badge](https://img.shields.io/badge/VSCode-blue)  ![Static Badge](https://img.shields.io/badge/Figma-black?logo=Figma)


# Image Aesthetics impact on Social Media Engagement
The repository explores various methodologies and technologies that contribute to the exploration if image aesthetics affect the engagement of the photo on social media.  
## Description
One of the most important and possible factors affecting user engagement on social networking sites is the aesthetic quality of photographs. The combination of **Deep Learning, Neural Networks** enables the development of an automated system for evaluating the effects and enagement predictability of images. This system provides dependable insights for optimizing content.
## Authors
Raj Navalakha - 23205373

Gargi Rajadnya - 23200711
## Data Extraction
Dataset obtained through web scraping (social media platform) using tools/website like Apify. Chosen type of data were food images. Extracted dataset contained around 1600 observations with different variables like:-
1. Shortcode - id for the dataset.
2. Likes count
3. Comments count
4. caption
5. hashtags
## Models used for prediction
1. Logistic Regression
2. SVC - Support Vector Classification
3. Decision Tree
4. Random Forest
5. XGBoost - Extreme Gradient Boosting
6. MLP - Multi-Layer Perceptron

Utilizing the machine learning and deep learning models mentioned above, we successfully improved the accuracy of all models in predicting the engagement dynamics for each image in the dataset. Additionally, the implementation of these models involved relevant Python libraries, including **Torch, Seaborn, SkLearn, PIL and TensorFlow**. (Evaluation metrics used are accuracy, F1 score, AUC-ROC curve and confsuion matrix for visualisations.)

## Tools 
Tools used to implement the project were Python, VSCode to integrate the scripts and Github to track the changes and commits made during the project.


## Project workflow

### Preprocessing: 
  


  
 
 ## Try out our codes ;)
  1) env_math_mod.yml is the YAML file used to create a python environment and to manage and keep a track of all the libraries. And conda env create —f env_math_mod.yml to create and launch the environemnt in your local machine.
  2) To run the scripts:
     i) 1_preprocessing.py: Data preprocessing
    ii) 2_data_expl_manip.py: Data exploration and manipulation
   iii) 3_model.py: Training and prediction.

