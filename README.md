[![python](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org) ![Static Badge](https://img.shields.io/badge/VSCode-blue)  ![Static Badge](https://img.shields.io/badge/Figma-black?logo=Figma)


# Image Aesthetics impact on Social Media Engagement
The repository explores various methodologies and technologies that contribute to the exploration if image aesthetics affect the engagement of the photo on social media.  
## Description
One of the most important and possible factors affecting user engagement on social networking sites is the aesthetic quality of photographs. The combination of **Deep Learning, Neural Networks** enables the development of an automated system for evaluating the effects and enagement predictability of images. This system provides dependable insights for optimizing content.
## Authors
[Raj Navalakha - 23205373](https://github.com/RajNavalakha29)

[Gargi Rajadnya - 23200711](https://github.com/gargirajadnya)

## Project workflow
![image](https://github.com/user-attachments/assets/ebeef4bb-bb61-4df3-a67a-66d49ff1eb2f)

### Data Extraction
Dataset obtained through web scraping (social media platform) using tools/website like Apify. Chosen type of data were food images. Extracted dataset contained around 1600 observations with different variables like:-
1. Shortcode - id for the dataset.
2. Likes count
3. Comments count
4. caption
5. hashtags

### Preprocessing (Script 1): 
Key libraries used:
1. Pandas for numerical data manipulation
2. NumPy for numerical operations
3. OpenCV and PIL for image processing

Preprocessing
1. Standardize column names
2. Check and address missing values (replace empty hashtags with 'NA')
3. Convert timestamps to a readable format
4. Clean text data (remove URLs, mentions, punctuation)
5. Calculate engagement metrics (combine likes and comments)
6. Filter data based on timestamps
7. Integrate processed image features with the original dataset

Image data is obtained from the URLs in the dataset. This data undergoes preprocessing and analysis with essential Python libraries. 
Different image attributes, including sharpness, exposure, colorfulness, vibrancy, symmetry, line detection, and composition, are assessed. These attributes are computed and recorded in a DataFrame, which is subsequently combined with the original dataset for thorough analysis.

### Exploration and Manipulation (Script 2):
Explored Target Variable,  and the correlation among numerical predictor variables was examined. VIF scores were calculated to assess multicollinearity among the predictors, and a pairplot was used to check for linear relationships between the target and predictor variables.

Feature engineering was carried out, including one-hot encoding for color names in the dataset. Box plots and the IQR method were employed to identify and remove outliers.

To address class imbalance, data sampling techniques were applied. Bootstrap sampling was used to balance class distribution, and upsampling was performed to increase the size of minority classes to match that of the majority class.

### Models used for prediction (Script 3)

The predictor variables were standardized, and the engagement metrics feature was split into a binary target variable to formulate a classification problem. The following models were used for prediction:

1. Logistic Regression
2. SVC (Support Vector Classification)
3. Decision Tree
4. Random Forest
5. XGBoost (Extreme Gradient Boosting)
6. MLP (Multi-Layer Perceptron)

For the MLP model, features were extracted from a trained CNN and combined with the previously engineered features in script 1 for further use. By applying the mentioned machine learning and deep learning models, the accuracy of all models in predicting engagement dynamics for each image in the dataset was improved. Relevant Python libraries used in the implementation included Torch, Seaborn, SkLearn, PIL, and TensorFlow. 

### Evaluation:
Evaluation metrics including accuracy, F1 score, AUC-ROC curve, and confusion matrix were employed to assess and visualize model performance. After reviewing all aspects and balancing the outcomes, XGBoost and MLP showed satisfactory results, though MLP displayed some indications of overfitting. The results could be further improved with higher-quality data, a larger dataset, and potentially exploring alternative methods for handling class imbalance, such as the Synthetic Minority Oversampling Technique (SMOTE).

#### Initial Regression problem results

RMSE score
![IMG_2591](https://github.com/user-attachments/assets/af3d4848-04eb-4c07-b606-4808e7646b76)

R^2 scores
![IMG_1305](https://github.com/user-attachments/assets/c6e088a6-1dcf-4410-8853-eb9653e45595)


#### Results after converting into a classification problem
#### Confusion matrices for the models
![IMG_2503](https://github.com/user-attachments/assets/407299c3-7219-4b8f-b09f-188a65b4f9ee)

![IMG_8766](https://github.com/user-attachments/assets/77af5b69-388e-46a7-acb0-94832b2afd53)

#### AUC-ROC curve for XGBoost and MLP
![IMG_6644](https://github.com/user-attachments/assets/988b0d5b-07fb-45bc-8b87-55556afe779f)

#### Learning curve of MLP
![IMG_4904](https://github.com/user-attachments/assets/0286e7bb-b0c2-4f20-9364-2dc668a5cfcc)


## Scope
This approach could evolve into predicting overall engagement metrics by considering all variables (like caption-sentiment analysis, caption language, no. of hashtags, etc) influencing engagement alongside image aesthetics.


## Tools 
Tools used to implement the project were Python, VSCode to integrate the scripts and Github to track the changes and commits made during the project.
 
## Try out our codes ;)
  1) env_math_mod.yml is the YAML file used to create a python environment and to manage and keep a track of all the libraries. And conda env create —f env_math_mod.yml to create and launch the environemnt in your local machine.
  2) To run the scripts:
     i) 1_preprocessing.py: Data preprocessing
    ii) 2_data_expl_manip.py: Data exploration and manipulation
   iii) 3_model.py: Training and prediction.

