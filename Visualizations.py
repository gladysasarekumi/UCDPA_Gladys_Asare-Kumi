# Import packages
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from Analysis import train_corr, train_corr_esg, feat_importances, train_features_df, predicted_returns, \
    predicted_returns_esg, monthly_returns_sp500_index


# VISUALIZATIONS

# 1. Heatmap of the feature-target correlations for the train data of the ESG and non-ESG filtered data

# Plot heatmap of the correlation matrix of the non-ESG data to visualize the correlation between target and features
sns.heatmap(train_corr, annot=True, annot_kws={"size": 6})  #create a heatmap and annotate the correlations on the map
plt.yticks(rotation=0, size=8)  # y axis tick label orientation and size
plt.xticks(rotation=90, size=8)  # x axis tick label orientation and size
plt.tight_layout()  # resize plot to fit in the window
plt.figure()  # create plot in a new window
plt.show()  #display the plot

# Plot heatmap of the correlation matrix for ESG data features to visualize the correlation between target and features
sns.heatmap(train_corr_esg, annot=True, annot_kws={"size": 6})  #create a heatmap and annotate the correlations on the map
plt.yticks(rotation=0, size=12)  # y axis tick label orientation and size
plt.xticks(rotation=90, size=12)  # x axis tick label orientation and size
plt.tight_layout()  # resize plot to fit in the window
plt.figure()  # create plot in a new window
plt.show()  #display the plot


# 2. Feature importances of random forest model after hyper parameter tuning

# Check feature importances to determine which features to keep or drop (only performed for non-ESG data)
sorted_idx = np.argsort(feat_importances)[::-1]  # Sort descending the indices of the importances
idx_rng = range(len(feat_importances))  # Determine the range of the importances for plotting
features = train_features_df.columns  # Get the list of features

# Create tick labels
lbls = np.array(features)[sorted_idx]
plt.bar(idx_rng, feat_importances[sorted_idx], tick_label=lbls)

# Rotate tick labels to vertical
plt.xticks(rotation=90)
plt.tight_layout()
plt.figure()
plt.show()  # show the plot


# 3. Comparing the predicted returns of the ESG-Filtered data and the non-ESG filtered data with the returns of the
# S&P500 index

# Plot the S&P500 index returns and the random forest predicted returns for both the ESG-filtered and non-ESG filtered
# data to visualize
plt.plot(predicted_returns, label='Predicted Returns', marker='o')
plt.plot(predicted_returns_esg, label='Predicted Returns ESG', marker='s')
plt.plot(monthly_returns_sp500_index, label='S&P500 Index Returns', marker='x')
plt.xlabel('Month')
plt.ylabel('Returns')
plt.xticks(rotation=90, size=12)  # fix ticklabel directions and size
plt.legend()
plt.tight_layout()
plt.figure()
plt.show()

