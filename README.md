[![python](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org) ![Static Badge](https://img.shields.io/badge/VSCode-blue)  ![Static Badge](https://img.shields.io/badge/Figma-black?logo=Figma)


# Image Aesthetics impact on Social Media Engagement
The repository explores various methodologies and technologies that contribute to the exploration if image aesthetics affect the engagement of the photo on social media.  
## Description
One of the most important and possible factors affecting user engagement on social networking sites is the aesthetic quality of photographs. The combination of **Deep Learning, Neural Networks** enables the development of an automated system for evaluating the effects and enagement predictability of images. This system provides dependable insights for optimizing content.
## Authors
[Raj Navalakha - 23205373](https://github.com/RajNavalakha29)

[Gargi Rajadnya - 23200711](https://github.com/gargirajadnya)

## Project workflow

### Data Extraction
Dataset obtained through web scraping (social media platform) using tools/website like Apify. Chosen type of data were food images. Extracted dataset contained around 1600 observations with different variables like:-
1. Shortcode - id for the dataset.
2. Likes count
3. Comments count
4. caption
5. hashtags

### Preprocessing: 
Key libraries used:
1. Pandas for numerical data manipulation
2. NumPy for numerical operations
3. OpenCV and PIL for image processing

1. Standardize column names
2. Check and address missing values (replace empty hashtags with 'NA')
3. Convert timestamps to a readable format
4. Clean text data (remove URLs, mentions, punctuation)
5. Calculate engagement metrics (combine likes and comments)
6. Filter data based on timestamps
7. Integrate processed image features with the original dataset

Image data is obtained from the URLs in the dataset. This data undergoes preprocessing and analysis with essential Python libraries. 
Different image attributes, including sharpness, exposure, colorfulness, vibrancy, symmetry, line detection, and composition, are assessed. These attributes are computed and recorded in a DataFrame, which is subsequently combined with the original dataset for thorough analysis.


### Exploration and Manipulation
Explored Target Variable,  and the correlation among numerical predictor variables was examined. VIF scores were calculated to assess multicollinearity among the predictors, and a pairplot was used to check for linear relationships between the target and predictor variables.

Feature engineering was carried out, including one-hot encoding for color names in the dataset. Box plots and the IQR method were employed to identify and remove outliers.

To address class imbalance, data sampling techniques were applied. Bootstrap sampling was used to balance class distribution, and upsampling was performed to increase the size of minority classes to match that of the majority class.

### Models used for prediction
1. Logistic Regression
2. SVC - Support Vector Classification
3. Decision Tree
4. Random Forest
5. XGBoost - Extreme Gradient Boosting
6. MLP - Multi-Layer Perceptron

Utilizing the machine learning and deep learning models mentioned above, we successfully improved the accuracy of all models in predicting the engagement dynamics for each image in the dataset. Additionally, the implementation of these models involved relevant Python libraries, including **Torch, Seaborn, SkLearn, PIL and TensorFlow**. (Evaluation metrics used are accuracy, F1 score, AUC-ROC curve and confsuion matrix for visualisations.)

## Tools 
Tools used to implement the project were Python, VSCode to integrate the scripts and Github to track the changes and commits made during the project.
 
## Try out our codes ;)
  1) env_math_mod.yml is the YAML file used to create a python environment and to manage and keep a track of all the libraries. And conda env create —f env_math_mod.yml to create and launch the environemnt in your local machine.
  2) To run the scripts:
     i) 1_preprocessing.py: Data preprocessing
    ii) 2_data_expl_manip.py: Data exploration and manipulation
   iii) 3_model.py: Training and prediction.

