#%%
#loading libraries
#basic
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


#
from scipy import stats
from sklearn.utils import resample
import statsmodels.api as sm
from scipy.stats import ttest_ind
from statsmodels.stats.outliers_influence import variance_inflation_factor

#plots
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt



#%%
#reading data
food_df = pd.read_csv('/Users/project/data/processed_data.csv')
food_df.head()

#%%
#check for missing values
missing_values = food_df.isnull().sum()
print(missing_values)

#%%
#replace NaN values in eng_met with 0
food_df['eng_met'] = np.nan_to_num(food_df['eng_met'], nan=0)

#%%
#let's drop unnecessary columns
cols_drop = ['node_comments_disabled', 'node_typename', 'node_id', 'raw_caption',
     'timestamp','node_edge_media_preview_like_count',
       'node_owner_id', 'node_thumbnail_src', 'node_thumbnail_resources_0_src',
       'node_thumbnail_resources_0_config_width',
       'node_thumbnail_resources_0_config_height',
       'node_thumbnail_resources_1_src',
       'node_thumbnail_resources_1_config_width',
       'node_thumbnail_resources_1_config_height',
       'node_thumbnail_resources_2_src',
       'node_thumbnail_resources_2_config_width',
       'node_thumbnail_resources_2_config_height',
       'node_thumbnail_resources_3_src',
       'node_thumbnail_resources_3_config_width',
       'node_thumbnail_resources_3_config_height',
       'node_thumbnail_resources_4_src',
       'node_thumbnail_resources_4_config_width',
       'node_thumbnail_resources_4_config_height', 'node_is_video',
       'node_accessibility_caption', 'node_product_type',
       'node_video_view_count', 'hashtags', 'caption','sharpn']

food_df = food_df.drop(columns = cols_drop)


#%%
#let's explore our target variable
# Histogram for eng_met
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(food_df['eng_met_base'], bins=30, color='blue', alpha=0.7)
plt.title('Distribution of eng_met')
plt.xlabel('eng_met')
plt.ylabel('Frequency')

# Histogram for eng_met_scaled
plt.subplot(1, 2, 2)
plt.hist(food_df['eng_met'], bins=30, color='green', alpha=0.7)
plt.title('Distribution of eng_met_scaled')
plt.xlabel('eng_met_scaled')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# %%
#specific columns for correlation and prediciton
num_f = food_df.select_dtypes(include=[np.number]).columns.tolist()

print("Numerical Features:", num_f)

col_int = ['dim_h', 'dim_w', 'brilliance', 'colorfulness', 'vibrancy', 'tint', 'definition', 'vignette', 'tone', 'depth', 'contrast', 'brightness', 'symmetry_score', 'center_score', 'eng_met'] 
corr_cols = ['dim_h', 'dim_w', 'brilliance', 'colorfulness', 'vibrancy', 'tint', 'definition', 'vignette', 'tone', 'depth', 'contrast', 'brightness', 'symmetry_score', 'center_score']


correlation_matrix = food_df[corr_cols].corr()

#colormap
mako_cmap = sns.color_palette("Greens", as_cmap=True)

#heatmap of correlation
plt.figure(figsize=(10, 8), facecolor='#DAD7CD')
sns.heatmap(correlation_matrix, annot=True, cmap=mako_cmap)
plt.title('Correlation Heatmap')
plt.show()


#scatter plot to check if there is any linear relationship between the target variable and predictors

#list
pred = ['dim_h', 'dim_w', 'brilliance', 'colorfulness', 'vibrancy', 'tint', 'definition', 'vignette', 'tone', 'depth', 'contrast', 'brightness', 'symmetry_score', 'center_score']
target = 'eng_met'

dark_green = "#2A3923" 
plot_bg_color = "#DAD7CD"

#initialize PairGrid
g = sns.PairGrid(food_df, y_vars=[target], x_vars=pred, height=2.5, aspect=1.0)

#map a scatterplot
g.map(sns.scatterplot, color=dark_green)

#regression lines to the plots
g.map(sns.regplot, scatter_kws={'color': dark_green, 'alpha': 0.5}, line_kws={'color': 'red'}, ci=None)

plt.gcf().set_facecolor(plot_bg_color) 

#titles
for ax, col in zip(g.axes.flat, pred):
    ax.set_title(col)

#plot
plt.show()

#%%
#let's check for multicollinearity and significant variables
#VIF scores for each feature
X = food_df[col_int].dropna()

#constant column for statsmodels
X = sm.add_constant(X)

#store VIF scores
vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

#drop the constant column VIF score
vif_data = vif_data.drop(vif_data[vif_data['Feature'] == 'const'].index)

high_vif_features = vif_data[vif_data['VIF'] > 10]
print(high_vif_features)

#%%
#feature engineering
#color names- one hot encoding
selected_df = food_df.copy()

# One-hot encode 'caption_lang' using 1 and 0
# caption_lang_encoded = pd.get_dummies(selected_df['caption_lang'], prefix='lang').astype(int)

#list of colors in 'color_names' to individual columns for one-hot encoding
color_names_exploded = selected_df['color_names'].apply(lambda x: x.strip("[]").replace("'", "").split(', ') if pd.notna(x) else [])

#top 3 dominant colors
top_colors_exploded = color_names_exploded.apply(lambda x: x[:3])

#unique colors from the top 3
unique_colors = set(color for sublist in top_colors_exploded for color in sublist)

#sorted list of unique colors
unique_colors_list = sorted(list(unique_colors))

#initialize DataFrame for color one-hot encoding
color_one_hot = pd.DataFrame(0, index=selected_df.index, columns=unique_colors_list)

# one-hot encoding
for i, colors in enumerate(top_colors_exploded):
    for color in colors:
        if color in color_one_hot.columns:
            color_one_hot.loc[i, color] = 1

#concatenate the one-hot encoded columns
food_df = pd.concat([selected_df
# , caption_lang_encoded
, color_one_hot], axis=1).drop([ 'color_names'
                                # ,'caption_lang'
                                , 'dominant_colors'], axis=1)

food_df.head()

#%%
model_df = food_df.copy()

# %%
outlier_df = model_df[['dim_h', 'dim_w', 'brilliance', 'colorfulness', 'vibrancy', 'tint', 'definition', 'vignette', 'tone', 'depth', 'contrast', 'brightness', 'symmetry_score', 'center_score'
                       , 'eng_met'
                       ]]
outlier_df.shape


#%%

#keep track of rows to remove
rows_to_remove = pd.Series([False] * len(model_df))

#keep track of the number of outliers per column
outlier_counts = {}

#iterate to detect outliers
numeric_columns = outlier_df.select_dtypes(include=['float64', 'int64']).columns

for column_name in numeric_columns:
    #Q1, Q3, and IQR
    Q1 = outlier_df[column_name].quantile(0.25)
    Q3 = outlier_df[column_name].quantile(0.75)
    IQR = Q3 - Q1

    #identify outliers
    is_outlier = (outlier_df[column_name] < (Q1 - 1.5 * IQR)) | (outlier_df[column_name] > (Q3 + 1.5 * IQR))
    
    #count outliers
    num_outliers = is_outlier.sum()
    outlier_counts[column_name] = num_outliers

    #update rows_to_remove 
    rows_to_remove = rows_to_remove | is_outlier

    #box plots
    plt.figure(figsize=(10, 6))
    plt.boxplot(outlier_df[column_name], vert=False)
    plt.title(f'Box plot for {column_name}')
    plt.xlabel(column_name)
    plt.show()

#remove the rows with any outliers
model_df = model_df[~rows_to_remove]

#number of outliers per column
print("Number of outliers per column:")
for column_name, count in outlier_counts.items():
    print(f"{column_name}: {count}")

#shape of data
print(f"Shape of DataFrame after removing outliers: {model_df.shape}")

#%%
#%%
#DATA SAMPLING - bootstrap sampling
#target number of rows
target_rows = 1000

#bootstrap sampling
bootstrapped_df = model_df.sample(n=target_rows, replace=True, random_state=1)

# bootstrap_sampled_data.to_csv('expanded_model_data.csv', index=False)

#shape of the new dataset
print("Shape of augmented dataset:", bootstrapped_df.shape)


#%%
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
original_class_counts = model_df_cleaned['eng_met'].value_counts()
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
    class_subset = model_df_cleaned[model_df_cleaned['eng_met'] == cls]
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
balanced_class_counts = model_df_balanced['eng_met'].value_counts()
print("\nClass distribution after bootstrapping:")
print(balanced_class_counts)

#shape
print(f"\nShape of the balanced dataset: {model_df_balanced.shape}")
print(f"Number of new observations added: {new_observations_count}")


#%%
# -------------------------------------------------------------------------
#save dataframes to train models according to needs
#original
food_df.to_csv("/Users/project/data/food_df.csv")

#no outliers
model_df.to_csv("/Users/project/data/model_df.csv")

#bootstrapping
bootstrapped_df.to_csv('/Users/project/data/bootstrapped_df.csv', index=False)

#outliers-preds, class imbalance- handling
model_df_balanced.to_csv("/Users/project/data/model_df_balanced.csv")


#%%