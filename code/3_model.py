#%%
#load libraries
#basic
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#class imbalance handling
from scipy import stats
from sklearn.utils import resample

#scaling
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


#splitting data
from sklearn.model_selection import train_test_split

#models
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from PIL import Image
import numpy as np
import os

#metrics
from sklearn.metrics import mean_squared_error,  r2_score, accuracy_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
import itertools
from sklearn.model_selection import learning_curve

#%%
#read model data
#without outlier/sampling
food_df = pd.read_csv('/Users/project/data/food_df.csv')


#%%
#create binary target variable
food_df['eng_met_bin'] = (food_df['eng_met_base'] > 100).astype(int)


# %%
#outliers and class imbalance

imb_df = food_df.copy()

#predictors list
predictors = ['dim_h', 'dim_w', 'brilliance', 'colorfulness', 'vibrancy', 
              'tint', 'definition', 'vignette', 'tone', 'depth', 
              'contrast', 'brightness', 'symmetry_score', 'center_score']

#shape of data before outlier removal
initial_shape = imb_df.shape
print(f"Shape of data before outlier removal: {initial_shape}")

#Z-scores to identify outliers
z_scores = np.abs(stats.zscore(imb_df[predictors]))

#number of outliers in each predictor
outliers_count = (z_scores > 3).sum(axis=0)
outliers_summary = pd.DataFrame({'Predictor': predictors, 'Outliers_Count': outliers_count})
print("\nNumber of outliers for each predictor:")
print(outliers_summary)

#remove outliers
model_df_cleaned = imb_df[(z_scores < 3).all(axis=1)]

#shape of data after outlier removal
final_shape = model_df_cleaned.shape
print(f"\nShape of data after outlier removal: {final_shape}")

#class distribution before bootstrapping
original_class_counts = model_df_cleaned['eng_met_bin'].value_counts()
print("\nClass distribution before bootstrapping:")
print(original_class_counts)

#handle class imbalance by bootstrapping minority classes
#find the size of the majority class
majority_class_size = original_class_counts.max()

#initialize a list to store the balanced dataframes
upsampled_dataframes = []
new_observations_count = 0

#bootstrap each class to bring them closer to balance
for cls in original_class_counts.index:
    class_subset = model_df_cleaned[model_df_cleaned['eng_met_bin'] == cls]
    current_size = len(class_subset)
    if current_size == majority_class_size:
        upsampled_dataframes.append(class_subset)
    else:
        #minority class
        upsampled_subset = resample(class_subset, 
                                    replace=True, 
                                    n_samples=majority_class_size,
                                    random_state=42)
        upsampled_dataframes.append(upsampled_subset)
        new_observations_count += len(upsampled_subset) - current_size

#combine: original and upsampled data
model_df_balanced = pd.concat(upsampled_dataframes)

#shuffle
model_df_balanced = model_df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

#class distribution after bootstrapping
balanced_class_counts = model_df_balanced['eng_met_bin'].value_counts()
print("\nClass distribution after bootstrapping:")
print(balanced_class_counts)

#shape
print(f"\nShape of the balanced dataset: {model_df_balanced.shape}")
print(f"Number of new observations added: {new_observations_count}")

#%%
#setting seed
np.random.seed(42)  
tf.random.set_seed(42)

#%%

#%%
#standardizing numerical columns
X = model_df_balanced[['dim_h', 'dim_w', 'brilliance', 'colorfulness', 'vibrancy', 'tint', 'definition', 'vignette', 'tone', 'depth', 'contrast', 'brightness', 'symmetry_score', 'center_score']]

y = model_df_balanced['eng_met_bin']

#numerical features
numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()

print("Numerical Features:", numerical_features)

#%%
#standardizing
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X[numerical_features])
X_normalized = pd.DataFrame(X_normalized, columns=numerical_features)

X_normalized.head()

#%%
#PCA
pca = PCA()
X_pca = pca.fit_transform(X_normalized)
explained_variance = pca.explained_variance_ratio_
print("Explained Variance Ratio:", explained_variance)

X_selected_pca = X_pca[:, :20]

# %%
#splitting data into train and test
# X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_selected_pca, y, test_size=0.3, random_state=42)

X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# %%
#initialize the linear regression model, train and make predictions
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_pca, y_train)
y_pred_lr = log_reg.predict(X_test_pca)

#evaluation metrics
accuracy_lr = accuracy_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr, average='weighted')

print('Multiple Linear Regression results:')
print("Accuracy:", round(accuracy_lr, 3))
print("F1 Score:", round(f1_lr, 3))

#%%
#initialize the SVC model, train and make predictions
svc = SVC(kernel='rbf', random_state=42)
svc.fit(X_train_pca, y_train)
y_pred_svc = svc.predict(X_test_pca)

#evaluation metrics
accuracy_svc = accuracy_score(y_test, y_pred_svc)
f1_svc = f1_score(y_test, y_pred_svc, average='weighted')

print('Support Vector Regression results:')
print("Accuracy:", round(accuracy_svc, 3))
print("F1 Score:", round(f1_svc, 3))

#%%
#initialize Decision Tree Regressor, train and make predictions
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train_pca, y_train)
y_pred_dt = dt_classifier.predict(X_test_pca)

#evaluation metrics
accuracy_dt = accuracy_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt, average='weighted')

print('Decision Tree Regression results:')
print("Accuracy:", round(accuracy_dt, 3))
print("F1 Score:", round(f1_dt, 3))

#%%
#initialize Random Forest, train and make predictions 
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30]
}

rf_classifier = RandomForestClassifier(random_state=42)
grid_search_rf = GridSearchCV(estimator=rf_classifier, param_grid=param_grid_rf, cv=5, scoring='f1_weighted', verbose=1, n_jobs=-1)
grid_search_rf.fit(X_train_pca, y_train)
best_params_rf = grid_search_rf.best_params_
best_rf_classifier = grid_search_rf.best_estimator_
y_pred_rf = best_rf_classifier.predict(X_test_pca)

#evaluation metrics
accuracy_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')

print('Random Forest results:')
print('Best Hyperparameters for Random Forest:', best_params_rf)
print("Accuracy:", round(accuracy_rf, 3))
print("F1 Score:", round(f1_rf, 3))

#%%
#initialize the XGBoost model, train and make predictions 
xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')
xgb_model.fit(X_train_pca, y_train)
y_pred_xgb = xgb_model.predict(X_test_pca)

#evaluation metrics
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb, average='weighted')

print('XGBoost results:')
print("Accuracy:", round(accuracy_xgb, 3))
print("F1 Score:", round(f1_xgb, 3))

#%%
#----------------------------------------------------------------------------------

#%%
#function to load and preprocess images
image_dir = "/Users/project/code/saved_images"

def preprocess_images(shortcodes, image_dir, target_size):
    processed_images = []
    for shortcode in shortcodes:
        #image path
        image_path = os.path.join(image_dir, f"{shortcode}.jpg")

        #image
        image = Image.open(image_path)

        #resize the image
        image = image.resize(target_size)

        #image to a numpy array and normalize pixel values to [0, 1]
        image_array = np.array(image) / 255.0  

        #single-channel (e.g., grayscale) images to RGB
        if image_array.ndim == 2:
            image_array = np.expand_dims(image_array, axis=-1)
            image_array = np.concatenate([image_array] * 3, axis=-1)

        #append the processed image to the list
        processed_images.append(image_array)

    #list of images to a NumPy array
    return np.array(processed_images)

shortcodes = model_df_balanced['shortcode'].values
image_arrays = preprocess_images(shortcodes, image_dir, target_size=(128, 128))


#%%
#CNN
def create_cnn_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(name='flatten_layer'),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

input_shape = (128, 128, 3)
cnn_model = create_cnn_model(input_shape)
cnn_model.summary()

#layer names to identify the correct Flatten layer name
for layer in cnn_model.layers:
    print(f"Layer name: {layer.name}")

#split the df for labels
y = model_df_balanced['eng_met_bin'].values

#split the data - training and testing
X_train_images, X_test_images, y_train, y_test = train_test_split(image_arrays, y, test_size=0.2, random_state=42)

#images are numpy arrays
X_train_images = np.array(X_train_images)
X_test_images = np.array(X_test_images)

#train model
cnn_model.fit(X_train_images, y_train, epochs=10, batch_size=32, validation_split=0.2)

#extract features using CNN
def extract_features_from_cnn(cnn_model, images):
    #create a model that ends with the last Flatten layer
    feature_extractor = Model(inputs=cnn_model.input, outputs=cnn_model.get_layer('flatten_layer').output)
    
    #shape
    print("Feature Extractor Output Shape:", feature_extractor.output_shape)
    
    #extract features
    features = feature_extractor.predict(images)
    return features

#features from CNN
X_train_features = extract_features_from_cnn(cnn_model, X_train_images)
X_test_features = extract_features_from_cnn(cnn_model, X_test_images)

print(f"Extracted features shape: {X_train_features.shape}")


#%%
#standardize the features
scaler = StandardScaler()
X_train_features = scaler.fit_transform(X_train_features)
X_test_features = scaler.transform(X_test_features)

#combining numerical features and cnn output features
X_train_combined = np.hstack((X_train_features, X_train_pca))
X_test_combined = np.hstack((X_test_features, X_test_pca))

#%%
#initialize MLP model, train and make predictions
mlp_model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
mlp_model.fit(X_train_combined, y_train)
y_pred_mlp = mlp_model.predict(X_test_combined)
y_pred_prob_mlp = mlp_model.predict_proba(X_test_combined)[:, 1]


#evaluation metrics
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
f1_mlp = f1_score(y_test, y_pred_mlp, average='weighted')

print('MLP model results:')
print("Accuracy:", round(accuracy_mlp, 3))
print("F1 Score:", round(f1_mlp, 3))

# %%
#----------------------------------------------------------------------------------------

#%% 
#plots
#function to plot confusion matrix
def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Greens):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    #normalize the confusion matrix
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    #plot the numbers on the matrix
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#confusion matrices for XGBoost, MLP, Random Forest, and Decision trees
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
cm_mlp = confusion_matrix(y_test, y_pred_mlp)
cm_rf = confusion_matrix(y_test, y_pred_rf)
cm_dt = confusion_matrix(y_test, y_pred_dt)
cm_lr = confusion_matrix(y_test, y_pred_lr)
cm_svc = confusion_matrix(y_test, y_pred_svc)

plt.figure(figsize=(18, 12), facecolor='#DAD7CD')

plt.subplot(2, 3, 1)
plot_confusion_matrix(cm_xgb, classes=['No', 'Post'], title='Confusion Matrix - XGBoost')

plt.subplot(2, 3, 2)
plot_confusion_matrix(cm_mlp, classes=['No', 'Post'], title='Confusion Matrix - MLP')

plt.subplot(2, 3, 3)
plot_confusion_matrix(cm_rf, classes=['No', 'Post'], title='Confusion Matrix - Random Forest')

plt.subplot(2, 3, 4)
plot_confusion_matrix(cm_dt, classes=['No', 'Post'], title='Confusion Matrix - Decision Tree')

plt.show()

plt.subplot(2, 3, 5)
plot_confusion_matrix(cm_lr, classes=['No', 'Post'], title='Confusion Matrix - Logistic Regression')

plt.subplot(2, 3, 6)
plot_confusion_matrix(cm_svc, classes=['No', 'Post'], title='Confusion Matrix - SVC')

plt.show()


# %%
#AUC-ROC Curve
#probability predictions for the positive class (class 1)
y_pred_prob_xgb = xgb_model.predict_proba(X_test_pca)[:, 1]
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_prob_xgb)
fpr_mlp, tpr_mlp, _ = roc_curve(y_test, y_pred_prob_mlp)

#AUC scores
auc_xgb = roc_auc_score(y_test, y_pred_prob_xgb)
auc_mlp = roc_auc_score(y_test, y_pred_prob_mlp)

plt.figure(figsize=(12, 8), facecolor='#DAD7CD')

#XGBoost ROC curve
plt.plot(fpr_xgb, tpr_xgb, color='darkgreen', linestyle='-', linewidth=2, marker='o', markersize=6, label=f'XGBoost (AUC = {auc_xgb:.3f})')

#MLP ROC curve
plt.plot(fpr_mlp, tpr_mlp, color='lightgreen', linestyle='--', linewidth=2, marker='s', markersize=6, label=f'MLP (AUC = {auc_mlp:.3f})')

#diagonal line
# plt.plot([0, 1], [0, 1], 'k--', lw=1)

#limits and labels
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('AUC-ROC Curve', fontsize=16)
plt.legend(loc='lower right', fontsize=12)
plt.grid(True)

#plot
plt.show()


#%%

#learning curve plots for MLP and Cnn
#function to plot learning curve
def plot_learning_curve(estimator, X, y, title="Learning Curves", cv=None, n_jobs=None, train_sizes=np.linspace(0.1, 1.0, 5)):
    plt.figure(figsize=(10, 6), facecolor='#DAD7CD')
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.grid()
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()

#MLP learning curve
plot_learning_curve(mlp_model, X_train_combined, y_train, title="Learning Curve - MLP")


# %%
history = cnn_model.fit(X_train_images, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

#CNN learning curve 
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Learning Curve - CNN')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#%%
# %%

#accuracy F1 plots
models = ['Logistic Regression', 'SVC', 'Decision Tree', 'Random Forest', 'XGBoost', 'MLP']
accuracy_scores = [accuracy_lr, accuracy_svc, accuracy_dt, accuracy_rf, accuracy_xgb, accuracy_mlp]
f1_scores = [f1_lr, f1_svc, f1_dt, f1_rf, f1_xgb, f1_mlp]
#plotting
plt.figure(figsize=(20, 8), facecolor='#DAD7CD')
bar_width = 0.4
indices = np.arange(len(models))

#accuracy bars
plt.barh(indices, accuracy_scores, height=bar_width, color='darkgreen', label='Accuracy')

#F1 scores bars
plt.barh(indices + bar_width, f1_scores, height=bar_width, color='lightseagreen', label='F1 Score')

#labels and title
plt.yticks(indices + bar_width / 2, models)
plt.xlabel('Score')
plt.title('Model Performance Comparison (Accuracy vs F1 Score)')
plt.xlim(0, 1) 
plt.legend()

for i, (acc, f1) in enumerate(zip(accuracy_scores, f1_scores)):
    plt.text(acc + 0.01, i, f'{acc:.3f}', va='center', color='darkgreen', fontsize=12)
    plt.text(f1 + 0.01, i + bar_width, f'{f1:.3f}', va='center', color='lightgreen', fontsize=12)

#plot
plt.show()

#%%
