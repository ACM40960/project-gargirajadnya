#%%
#loading libraries
#basic
import pandas as pd
import numpy as np
import os

from statsmodels.stats.outliers_influence import variance_inflation_factor

#standardization
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


#splitting data
from sklearn.model_selection import train_test_split

#models
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout

#metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score, confusion_matrix, roc_curve, roc_auc_score

#plots
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

#%%
#read model data
#without outlier/sampling
food_df = pd.read_csv('/Users/gargirajadnya/Documents/Academic/UCD/Trimester 3/Math Modeling/Engagement_dynamics/data/food_df.csv')

#without outliers
model_df = pd.read_csv('/Users/gargirajadnya/Documents/Academic/UCD/Trimester 3/Math Modeling/Engagement_dynamics/data/model_df.csv')

#whole bootstrapping sampling
bootstrapped_df = pd.read_csv('/Users/gargirajadnya/Documents/Academic/UCD/Trimester 3/Math Modeling/Engagement_dynamics/data/bootstrapped_df.csv')

#class imbalance
model_df_bal = pd.read_csv('/Users/gargirajadnya/Documents/Academic/UCD/Trimester 3/Math Modeling/Engagement_dynamics/data/model_df_balanced.csv')


#%%
#------------------------------------------------------------------------
np.random.seed(42)  
tf.random.set_seed(42)

#%%
#standardizing numerical columns
X = model_df_bal[['dim_h', 'dim_w', 'brilliance', 'colorfulness', 'vibrancy', 'tint', 'definition', 'vignette', 'tone', 'depth', 'contrast', 'brightness', 'symmetry_score', 'center_score']]

y = model_df_bal['eng_met']

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
# B = X[numerical_features].dropna()

# #constant column for statsmodels
# Z = sm.add_constant(B)

# #store VIF scores
# vif_data = pd.DataFrame()
# vif_data['Feature'] = Z.columns
# vif_data['VIF'] = [variance_inflation_factor(Z.values, i) for i in range(Z.shape[1])]

# #drop the constant column VIF score
# vif_data = vif_data.drop(vif_data[vif_data['Feature'] == 'const'].index)

# vif_data[vif_data['VIF'] > 10]['Feature']


#%%
#PCA
pca = PCA()
X_pca = pca.fit_transform(X_normalized)
explained_variance = pca.explained_variance_ratio_
print("Explained Variance Ratio:", explained_variance)

X_selected_pca = X_pca[:, :20]

# %%
#splitting data into train and test
X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_selected_pca, y, test_size=0.3, random_state=42)

# X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# %%
#initialize the linear regression model
lin_reg = LinearRegression()
#train the model
lin_reg.fit(X_train_pca, y_train)
#predictions
y_pred_lr = lin_reg.predict(X_test_pca)

#evaluation metrics
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print('Multiple Linear Regression results:')
print("Mean Squared Error:", mse_lr)
print("RMSE:", rmse_lr)
print("Mean Absolute Error:", mae_lr)
print("R-squared:", r2_lr)

#%%
#initialize the SVR model
svr = SVR(kernel='rbf'
        #   , gamma = 0.15
          )
#train model
svr.fit(X_train_pca, y_train)
#predictions
y_pred_svr = svr.predict(X_test_pca)

#evaluation metrics
mse_svr = mean_squared_error(y_test, y_pred_svr)
rmse_svr = np.sqrt(mean_squared_error(y_test, y_pred_svr))
mae_svr = mean_absolute_error(y_test, y_pred_svr)
r2_svr = r2_score(y_test, y_pred_svr)

print('Support Vector Regression results:')
print("Mean Squared Error:", mse_svr)
print("RMSE:", rmse_svr)
print("Mean Absolute Error:", mae_svr)
print("R-squared:", r2_svr)


#%%
#initialize Decision Tree Regressor
dt_regressor = DecisionTreeRegressor(max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42)

#train model
dt_regressor.fit(X_train_pca, y_train)

#predictions
y_pred_dt = dt_regressor.predict(X_test_pca)

#evaluation metrics
mse_dt = mean_squared_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))
mae_dt = mean_absolute_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

print('Decision Tree Regression results:')
print("Mean Squared Error:", mse_dt)
print("RMSE:", rmse_dt)
print("Mean Absolute Error:", mae_dt)
print("R-squared:", r2_dt)

#%%
#initialize Random Forest 
rf = RandomForestRegressor(n_estimators=100, random_state=42)

#train model
rf.fit(X_train_pca, y_train)
#predictions
y_pred_rf = rf.predict(X_test_pca)

#evaluation metrics
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print('Random Forest results:')
print("Mean Squared Error:", mse_rf)
print("RMSE:", rmse_rf)
print("Mean Absolute Error:", mae_rf)
print("R-squared:", r2_rf)

#%%
#initialize the XGBoost model
model = xgb.XGBRegressor()
#train model
model.fit(X_train_pca, y_train)
#predictions
y_pred_xgb = model.predict(X_test_pca)

#evaluation metrics
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print('XGBoost results:')
print("Mean Squared Error:", mse_xgb)
print("RMSE:", rmse_xgb)
print("Mean Absolute Error:", mae_xgb)
print("R-squared:", r2_xgb)

# %%
#set seed for TensorFlow
tf.random.set_seed(42)
tf.config.experimental.enable_op_determinism()

#feature dimensions
input_dim = X_train_pca.shape[1]  

#MLP model
model = Sequential([
    #input layer
    Dense(128, activation='relu', input_shape=(input_dim,)),
    Dropout(0.2), 
    
    #hidden layers
    Dense(64, activation='relu'),
    Dropout(0.2),
    
    Dense(32, activation='relu'),
    Dropout(0.2),
    
    #output layer
    Dense(1) 
])

#compile the model
model.compile(optimizer=Adam(learning_rate=0.01),  
              loss='mean_squared_error',
              metrics=['mean_absolute_error'])

# model.compile(optimizer='adam',
#               loss='mean_squared_error',
#               metrics=['mean_absolute_error'])

from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
history = model.fit(X_train_pca, y_train, 
                    epochs=200, 
                    batch_size=32, 
                    validation_data=(X_test_pca, y_test),
                    callbacks=[early_stopping])

#predictions
y_pred_mlp = model.predict(X_test_pca)
#reshape 
y_pred_mlp_f = y_pred_mlp.flatten()


#evaluation metrics
mse_mlp = mean_squared_error(y_test, y_pred_mlp)
rmse_mlp = np.sqrt(mean_squared_error(y_test, y_pred_mlp))
mae_mlp = mean_absolute_error(y_test, y_pred_mlp)
r2_mlp = r2_score(y_test, y_pred_mlp)

print('MLP Regression results:')
print("Mean Squared Error:", mse_mlp)
print("RMSE:", rmse_mlp)
print("Mean Absolute Error:", mae_mlp)
print("R-squared:", r2_mlp)

# %%

#--------------------------------------------------------------------------------

#%%
#plots
#residuals df
residuals_df = pd.DataFrame({
    'Linear Regression': y_test - y_pred_lr,
    'SVR': y_test - y_pred_svr,
    'Decision Tree': y_test - y_pred_dt,
    'Random Forest': y_test - y_pred_rf,
    'XGBoost': y_test - y_pred_xgb,
    'MLP': y_test - y_pred_mlp_f
})

#color palette
custom_palette = ['#783D19', '#C4661F', '#95714F', '#C7AF94', '#8C916C', '#ACB087']

#Residuals Distribution
plt.figure(figsize=(12, 8))
sns.set(style="whitegrid")
sns.boxplot(data=residuals_df, palette=custom_palette)

plt.title('Residuals the for the Models', fontsize=20, fontweight='bold', color='#4A4A4A', pad=20)
plt.ylabel('Residuals', fontsize=16, fontweight='bold', color='#4A4A4A')
plt.xticks(rotation=45, fontsize=14, color='#4A4A4A', fontweight='bold')
plt.yticks(fontsize=14, color='#4A4A4A', fontweight='bold')


#make plot fancy
plt.grid(True, which='both', linestyle='--', linewidth=0.7, color='gray', alpha=0.7)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(1.2)
plt.gca().spines['left'].set_color('#4A4A4A')
plt.gca().spines['bottom'].set_linewidth(1.2)
plt.gca().spines['bottom'].set_color('#4A4A4A')

plt.show()


#---------------------------------------------------------------------------------
#metrics df
metrics_df = pd.DataFrame({
    'Model': ['Linear Regression', 'SVR', 'Decision Tree', 'Random Forest', 'XGBoost', 'MLP'],
    'R^2': [r2_lr, r2_svr, r2_dt, r2_rf, r2_xgb, r2_mlp],
    'RMSE': [rmse_lr, rmse_svr, rmse_dt, rmse_rf, rmse_xgb, rmse_mlp]
})

#----------------------------------------------------------------------------
#plot R^2 values
plt.figure(figsize=(12, 8))
sns.set(style="whitegrid")

#barplot with a custom palette
barplot = sns.barplot(x='Model', y='R^2', data=metrics_df, palette=custom_palette, edgecolor="black")

#make plot fancy
plt.title('R^2 Scores for Different Models', fontsize=20, fontweight='bold', color='#4A4A4A', pad=20)
plt.ylabel('R^2 Score', fontsize=16, fontweight='bold', color='#4A4A4A')
plt.xticks(rotation=45, fontsize=14, color='#4A4A4A', fontweight='bold')
plt.yticks(fontsize=14, color='#4A4A4A', fontweight='bold')

#y-axis limits adjusting
# plt.ylim(-1, 0.5)  

plt.grid(True, which='both', linestyle='--', linewidth=0.7, color='gray', alpha=0.7)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(1.2)
plt.gca().spines['left'].set_color('#4A4A4A')
plt.gca().spines['bottom'].set_linewidth(1.2)
plt.gca().spines['bottom'].set_color('#4A4A4A')

for index, value in enumerate(metrics_df['R^2']):
    text = plt.text(index, value + 0.02, f'{value:.2f}', ha='center', fontsize=12, fontweight='bold', color='#4A4A4A',
                    bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3'))
    #shadow effect
    text.set_path_effects([
        path_effects.withStroke(linewidth=1, foreground='gray', alpha=0.5),
        path_effects.Normal()
    ])

plt.show()

#----------------------------------------------------------------------------

#rmse
#color palette
custom_palette = ['#783D19', '#C4661F', '#95714F', '#C7AF94', '#8C916C', '#ACB087']

#plot size
plt.figure(figsize=(12, 8))
sns.set(style="whitegrid")

#barplot
barplot = sns.barplot(x='Model', y='RMSE', data=metrics_df, palette=custom_palette, edgecolor="black")

#mak eplot fancy
plt.title('RMSE for the Models', fontsize=20, fontweight='bold', color='#4A4A4A', pad=20)
plt.ylabel('RMSE', fontsize=16, fontweight='bold', color='#4A4A4A')
plt.xticks(rotation=45, fontsize=14, color='#4A4A4A', fontweight='bold')
plt.yticks(fontsize=14, color='#4A4A4A', fontweight='bold')

#y-axis limits
min_rmse = min(metrics_df['RMSE']) - 0.2
max_rmse = max(metrics_df['RMSE']) + 0.2
plt.ylim(min_rmse, max_rmse)

plt.grid(True, which='both', linestyle='--', linewidth=0.7, color='gray', alpha=0.7)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(1.2)
plt.gca().spines['left'].set_color('#4A4A4A')
plt.gca().spines['bottom'].set_linewidth(1.2)
plt.gca().spines['bottom'].set_color('#4A4A4A')

for index, value in enumerate(metrics_df['RMSE']):
    text = plt.text(index, value + 0.05, f'{value:.2f}', ha='center', fontsize=12, fontweight='bold', color='#4A4A4A',
                    bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3'))
    text.set_path_effects([
        path_effects.withStroke(linewidth=1, foreground='gray', alpha=0.5),
        path_effects.Normal()
    ])

#plot
plt.show()

#%%
#--------------------------------------------------------------------------------------------------------------------
#%%
#standardizing numerical columns
# model_df['eng_met'] = model_df['eng_met'].replace([np.inf, -np.inf], 0)

X = bootstrapped_df[['dim_h', 'dim_w', 'brilliance', 'colorfulness', 'vibrancy', 'tint', 'definition', 'vignette', 'tone', 'depth', 'contrast', 'brightness', 'symmetry_score', 'center_score']]

y = bootstrapped_df['eng_met']

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
# B = X[numerical_features].dropna()

# # Add a constant column for statsmodels
# Z = sm.add_constant(B)

# #store VIF scores
# vif_data = pd.DataFrame()
# vif_data['Feature'] = Z.columns
# vif_data['VIF'] = [variance_inflation_factor(Z.values, i) for i in range(Z.shape[1])]

# #drop the constant column VIF score
# vif_data = vif_data.drop(vif_data[vif_data['Feature'] == 'const'].index)

# vif_data[vif_data['VIF'] > 10]['Feature']


#%%
# #PCA
pca = PCA()
X_pca = pca.fit_transform(X_normalized)
explained_variance = pca.explained_variance_ratio_
print("Explained Variance Ratio:", explained_variance)

X_selected_pca = X_pca[:, :20]

# %%
#splitting into train and test datasets
# X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_selected_pca, y, test_size=0.3, random_state=42)

X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# %%
#initialize the linear regression model
lin_reg = LinearRegression()

#train model
lin_reg.fit(X_train_pca, y_train)

# Make predictions
y_pred_lr = lin_reg.predict(X_test_pca)

# Evaluate the model
mse_lr = mean_squared_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
mae_lr = mean_absolute_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print('Multiple Linear Regression results:')
print("Mean Squared Error:", mse_lr)
print("RMSE:", rmse_lr)
print("Mean Absolute Error:", mae_lr)
print("R-squared:", r2_lr)

#%%
svr = SVR(kernel='rbf'
          , gamma = 0.15
         )

#train model
svr.fit(X_train_pca, y_train)

y_pred_svr = svr.predict(X_test_pca)

mse_svr = mean_squared_error(y_test, y_pred_svr)
rmse_svr = np.sqrt(mean_squared_error(y_test, y_pred_svr))
mae_svr = mean_absolute_error(y_test, y_pred_svr)
r2_svr = r2_score(y_test, y_pred_svr)

print('Support Vector Regression results:')
print("Mean Squared Error:", mse_svr)
print("RMSE:", rmse_svr)
print("Mean Absolute Error:", mae_svr)
print("R-squared:", r2_svr)



#%%
# Initialize Decision Tree Regressor
dt_regressor = DecisionTreeRegressor(max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42)

#train model
dt_regressor.fit(X_train_pca, y_train)
#predictions
y_pred_dt = dt_regressor.predict(X_test_pca)
#evaluation metrics
mse_dt = mean_squared_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))
mae_dt = mean_absolute_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

# Print the results
print('Decision Tree Regression results:')
print("Mean Squared Error:", mse_dt)
print("RMSE:", rmse_dt)
print("Mean Absolute Error:", mae_dt)
print("R-squared:", r2_dt)

#%%
#initialize Random Forest 
rf = RandomForestRegressor(n_estimators=100, random_state=42)

#train model
rf.fit(X_train_pca, y_train)
#predictions
y_pred_rf = rf.predict(X_test_pca)
#evaluation metrics
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print('Random Forest results:')
print("Mean Squared Error:", mse_rf)
print("RMSE:", rmse_rf)
print("Mean Absolute Error:", mae_rf)
print("R-squared:", r2_rf)

#%%
#initialize the XGBoost model
model = xgb.XGBRegressor()

#train model
model.fit(X_train_pca, y_train)
#predictions
y_pred_xgb = model.predict(X_test_pca)
#evaluation metrics
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print('XGBoost results:')
print("Mean Squared Error:", mse_xgb)
print("RMSE:", rmse_xgb)
print("Mean Absolute Error:", mae_xgb)
print("R-squared:", r2_xgb)

#%%
#CNN
# # Example image dimensions
# img_height, img_width = 128, 128  # Adjust based on your images
# num_channels = 3  # RGB images

# #CNN model
# model = Sequential([
#     #convolutional layer
#     Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, num_channels)),
#     MaxPooling2D(pool_size=(2, 2)),
    
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),
    
#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPooling2D(pool_size=(2, 2)),
    
#     #flatten the output from the convolutional layers
#     Flatten(),
    
#     #fully connected layers
#     Dense(128, activation='relu'),
#     Dense(1)  # Output layer for regression
# ])

# #compile the model
# model.compile(optimizer=Adam(learning_rate=0.001),
#               loss='mean_squared_error',
#               metrics=['mean_absolute_error'])

# #train model
# history = model.fit(X_train_pca, y_train, 
#                     epochs=10,  # Adjust number of epochs as needed
#                     batch_size=32,  # Adjust batch size as needed
#                     validation_data=(X_test_pca, y_test))

# #predictions
# y_pred_cnn = model.predict(X_test_pca)

# #evaluation metrics
# mse_cnn = mean_squared_error(y_test, y_pred_cnn)
# mae_cnn = mean_absolute_error(y_test, y_pred_cnn)
# r2_cnn = r2_score(y_test, y_pred_cnn)

# print('CNN Regression results:')
# print("Mean Squared Error:", mse_cnn)
# print("Mean Absolute Error:", mae_cnn)
# print("R-squared:", r2_cnn)

# %%
#MLP
#feature dimensions
input_dim = X_train_pca.shape[1]  # Number of features

#MLP model
model = Sequential([
    #input layer
    Dense(128, activation='relu', input_shape=(input_dim,)),
    Dropout(0.2),  # Dropout layer to prevent overfitting
    
    #hidden layers
    Dense(64, activation='relu'),
    Dropout(0.2),
    
    Dense(32, activation='relu'),
    Dropout(0.2),
    
    #output layer
    Dense(1)
])

#compile the model
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mean_absolute_error'])

#train model
history = model.fit(X_train_pca, y_train, 
                    #hyperparameter tuning
                    epochs=500,
                    batch_size=32,
                    validation_data=(X_test_pca, y_test))

#predictions
y_pred_mlp = model.predict(X_test_pca)
y_pred_mlp_f = y_pred_mlp.flatten()

#evaluation metrics
mse_mlp = mean_squared_error(y_test, y_pred_mlp)
rmse_mlp = np.sqrt(mean_squared_error(y_test, y_pred_mlp))
mae_mlp = mean_absolute_error(y_test, y_pred_mlp)
r2_mlp = r2_score(y_test, y_pred_mlp)

print('MLP Regression results:')
print("Mean Squared Error:", mse_mlp)
print("RMSE:", rmse_mlp)
print("Mean Absolute Error:", mae_mlp)
print("R-squared:", r2_mlp)


#%%
#plots
#residuals
residuals_df = pd.DataFrame({
    'Linear Regression': y_test - y_pred_lr,
    'SVR': y_test - y_pred_svr,
    'Decision Tree': y_test - y_pred_dt,
    'Random Forest': y_test - y_pred_rf,
    'XGBoost': y_test - y_pred_xgb,
    'MLP': y_test - y_pred_mlp_f
})

#color palette
custom_palette = ['#C4661F', '#95714F', '#8C916C', '#ACB087', '#783D19', '#C7AF94']

#plot Residuals Distribution
plt.figure(figsize=(12, 8))
sns.set(style="whitegrid")
sns.boxplot(data=residuals_df, palette=custom_palette)

plt.title('Residuals Distribution for the Models', fontsize=20, fontweight='bold', color='#4A4A4A', pad=20)
plt.ylabel('Residuals', fontsize=16, fontweight='bold', color='#4A4A4A')
plt.xticks(rotation=45, fontsize=14, color='#4A4A4A', fontweight='bold')
plt.yticks(fontsize=14, color='#4A4A4A', fontweight='bold')


#make plot fancy
plt.grid(True, which='both', linestyle='--', linewidth=0.7, color='gray', alpha=0.7)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(1.2)
plt.gca().spines['left'].set_color('#4A4A4A')
plt.gca().spines['bottom'].set_linewidth(1.2)
plt.gca().spines['bottom'].set_color('#4A4A4A')

plt.show()


#---------------------------------------------------------------------------------
#metrics df
metrics_df = pd.DataFrame({
    'Model': ['Linear Regression', 'SVR', 'Decision Tree', 'Random Forest', 'XGBoost', 'MLP'],
    'R^2': [r2_lr, r2_svr, r2_dt, r2_rf, r2_xgb, r2_mlp],
    'RMSE': [rmse_lr, rmse_svr, rmse_dt, rmse_rf, rmse_xgb, rmse_mlp]
})

#----------------------------------------------------------------------------
#R^2
#size
plt.figure(figsize=(12, 8))
sns.set(style="whitegrid")

#barplot 
barplot = sns.barplot(x='Model', y='R^2', data=metrics_df, palette=custom_palette, edgecolor="black")

#make plot fancy
plt.title('R^2 Scores for Different Models', fontsize=20, fontweight='bold', color='#4A4A4A', pad=20)
plt.ylabel('R^2 Score', fontsize=16, fontweight='bold', color='#4A4A4A')
plt.xticks(rotation=45, fontsize=14, color='#4A4A4A', fontweight='bold')
plt.yticks(fontsize=14, color='#4A4A4A', fontweight='bold')

plt.grid(True, which='both', linestyle='--', linewidth=0.7, color='gray', alpha=0.7)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(1.2)
plt.gca().spines['left'].set_color('#4A4A4A')
plt.gca().spines['bottom'].set_linewidth(1.2)
plt.gca().spines['bottom'].set_color('#4A4A4A')

for index, value in enumerate(metrics_df['R^2']):
    text = plt.text(index, value + 0.02, f'{value:.2f}', ha='center', fontsize=12, fontweight='bold', color='#4A4A4A',
                    bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3'))
    text.set_path_effects([
        path_effects.withStroke(linewidth=1, foreground='gray', alpha=0.5),
        path_effects.Normal()
    ])

plt.show()

#----------------------------------------------------------------------------

#rmse
#color palette
custom_palette = ['#783D19', '#C4661F', '#95714F', '#C7AF94', '#8C916C', '#ACB087']

#size
plt.figure(figsize=(12, 8))
sns.set(style="whitegrid")

#barplot 
barplot = sns.barplot(x='Model', y='RMSE', data=metrics_df, palette=custom_palette, edgecolor="black")

#make plot fancy
plt.title('RMSE for the Models', fontsize=20, fontweight='bold', color='#4A4A4A', pad=20)
plt.ylabel('RMSE', fontsize=16, fontweight='bold', color='#4A4A4A')
plt.xticks(rotation=45, fontsize=14, color='#4A4A4A', fontweight='bold')
plt.yticks(fontsize=14, color='#4A4A4A', fontweight='bold')

#adjust the y-axis limits
min_rmse = min(metrics_df['RMSE']) - 0.2
max_rmse = max(metrics_df['RMSE']) + 0.2
plt.ylim(min_rmse, max_rmse)

plt.grid(True, which='both', linestyle='--', linewidth=0.7, color='gray', alpha=0.7)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(1.2)
plt.gca().spines['left'].set_color('#4A4A4A')
plt.gca().spines['bottom'].set_linewidth(1.2)
plt.gca().spines['bottom'].set_color('#4A4A4A')

for index, value in enumerate(metrics_df['RMSE']):
    text = plt.text(index, value + 0.05, f'{value:.2f}', ha='center', fontsize=12, fontweight='bold', color='#4A4A4A',
                    bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.3'))
    text.set_path_effects([
        path_effects.withStroke(linewidth=1, foreground='gray', alpha=0.5),
        path_effects.Normal()
    ])

#plot
plt.show()

#%%

#%%
#function to load and preprocess images
image_dir = "/Users/gargirajadnya/Documents/Academic/UCD/Trimester 3/Math Modeling/Engagement_dynamics/code/saved_images"

def preprocess_images(shortcodes, image_dir, target_size):
    processed_images = []
    for shortcode in shortcodes:
        # Construct the image path
        image_path = os.path.join(image_dir, f"{shortcode}.jpg")

        # Load the image
        image = Image.open(image_path)

        # Resize the image
        image = image.resize(target_size)

        # Convert the image to a numpy array and normalize pixel values to [0, 1]
        image_array = np.array(image) / 255.0  

        # If the image has only one channel (e.g., grayscale), convert it to RGB
        if image_array.ndim == 2:
            image_array = np.expand_dims(image_array, axis=-1)
            image_array = np.concatenate([image_array] * 3, axis=-1)  # Convert to RGB

        # Append the processed image to the list
        processed_images.append(image_array)

    # Convert the list of images to a NumPy array
    return np.array(processed_images)

# Example usage
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

# Print layer names to identify the correct Flatten layer name
for layer in cnn_model.layers:
    print(f"Layer name: {layer.name}")

#split the df for labels
y = model_df_balanced['eng_met_bin'].values

#split the data - training and testing
X_train_images, X_test_images, y_train, y_test = train_test_split(image_arrays, y, test_size=0.2, random_state=42)

# Ensure images are numpy arrays
X_train_images = np.array(X_train_images)
X_test_images = np.array(X_test_images)

# Train the CNN model
cnn_model.fit(X_train_images, y_train, epochs=10, batch_size=32, validation_split=0.2)

#extract features using CNN
def extract_features_from_cnn(cnn_model, images):
    #create a model that ends with the last Flatten layer
    feature_extractor = Model(inputs=cnn_model.input, outputs=cnn_model.get_layer('flatten_layer').output)
    
    # Print the shape of the intermediate features
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
