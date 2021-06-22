# HYPOTHESIS TESTING

# Import the necessary modules
import pandas as pd
import numpy as np
# import the below functions and variables from main.py
from Analysis import sp500_monthlyreturns, sp500esg_monthlyreturns, sp500_index_monthly_returns, sp500_cov, sp500esg_cov, \
    rfr_df_mth_avg, test_features_df, train_features_df, train_features_esg_df, test_features_esg_df, test_features, \
    train_features, test_features_esg, train_features_esg, get_target, calculate_hypothetical_portfolio_returns, \
    get_sharpe, random_forest, besthyp, esg_besthyp


# Define functions to be used
def perm_reps_mean_diff(x1, x2, size=1):
    """Generates permutation replicates and computes the differences of the sample means
    Args:
    x1 (array): The first dataset to be compared
    x2 (array): The second dataset to be compared
    size(optional, int): The number of permutation samples to be generated
    Returns: array
    """
    # Initialize array of replicates: perm_replicates
    perms = np.empty(size)

    for y in range(size):
        # Generate permutation sample
        x = np.concatenate([x1, x2])
        perm = np.random.permutation(x)
        s1 = perm[:len(x1)]  # get sample 1
        s2 = perm[len(x1):]    # get sample 2

        # Compute the difference in the means of the two samples
        perms[y] = np.mean(s1) - np.mean(s2)

    return perms


# Compute the predicted overall returns of the hypothetical portfolio to use to test the hypothesis
esg_returns = []  # initialize list to hold the hypothetical portfolio returns for the ESG_filtered data
nonesg_returns = []  # initialize list to hold the hypothetical portfolio returns for the non-ESG_filtered data

for f in range(10):  # run 10 simulations of the hypothetical portfolio to be used in the hypothesis test

    # Calculate portfolio returns and volatility for each date for the non-ESG filtered data
    test_ret_vol = []  # initialize empty list to hold the portfolio returns and volatility
    test_port_weights = []   # initialize empty list  to hold the portfolio weights
    rfr_list = []  # initialize empty list to hold the risk-free rate
    x = len(sp500_monthlyreturns.columns)  # get number of stocks in the monthly returns dataframe
    # get portfolio performances for each month
    for y in sp500_cov:  # loop through the list of covariances
        covar = y[1]
        ret_temp, vol_temp, wts_temp = [], [], []  # initialize empty lists to hold the calculated values at each iteration
        rfr = rfr_df_mth_avg[rfr_df_mth_avg.index.strftime('%Y-%m') == str(y[0])][0]  # obtain the risk free rate for the period y
        for portfolio in range(10):  # run 10 simulations of the portfolio for each date period
            wts = np.random.random(x)  # generate random weights for each security in the portfolio
            wts = wts/np.sum(wts)  # normalize so all weights sum to 1
            # calculate the returns of the simulated portfolio
            returns = sp500_monthlyreturns[sp500_monthlyreturns.index.strftime('%Y-%m') == str(y[0])].values
            # rotate the returns list to have the same shape as the weights list so the dot product can be computed
            returns = returns.reshape(wts.shape)
            ret = np.dot(wts, returns)  # calculate the return of the portfolio
            vol = np.sqrt(np.dot(wts.T, np.dot(covar, wts)))  # compute the portfolio volatility
            ret_temp.append(ret)  # append the simulated portfolio return to a temporary list to hold the simulations
            vol_temp.append(vol)  # append the simulated portfolio volatility to a temporary list to hold the simulations
            wts_temp.append(wts)  # append the simulated portfolio weight to a temporary list to hold the simulations
        rfr_list.append([y[0], rfr])  # append the risk-free rate for that month period to its list
        # add the period, portfolio returns list and portfolio volatility list to the final returns and volatility list
        test_ret_vol.append([y[0], [ret_temp], [vol_temp]])
        # add the period and portfolio weights list to the final weights list
        test_port_weights.append([y[0], wts_temp])


    # Calculate portfolio returns and volatility for each date for the ESG filtered data
    rdm_wt_port_ret_vol_esg = []  # initialize empty list to hold the portfolio returns and volatility
    rdm_wt_port_weights_esg = []   # initialize empty list to hold the portfolio weights
    rfr_list_esg = []  # initialize empty list to hold the risk-free rate
    x = len(sp500esg_monthlyreturns.columns)  # get number of stocks in monthly returns dataframe
    # get portfolio performances for each month
    for y in sp500esg_cov:
        covar = y[1]
        ret_temp, vol_temp, wts_temp = [], [], []  # initialize empty lists to hold the calculated values at each iteration
        rfr = rfr_df_mth_avg[rfr_df_mth_avg.index.strftime('%Y-%m') == str(y[0])][0]  # obtain the risk free rate for the period y
        for portfolio in range(10):  # run 10 simulations of the portfolio for each date period
            wts = np.random.random(x)  # generate random weights for each security in the portfolio
            wts = wts/np.sum(wts)  # normalize so all weights sum to 1
            # calculate the returns of the simulated portfolio
            returns = sp500esg_monthlyreturns[sp500esg_monthlyreturns.index.strftime('%Y-%m')== str(y[0])].values
            # rotate the returns list to have the same shape as the weights list so the dot product can be computed
            returns = returns.reshape(wts.shape)
            ret = np.dot(wts, returns)  # calculate the return of the portfolio
            vol = np.sqrt(np.dot(wts.T, np.dot(covar, wts)))  # compute the portfolio volatility
            ret_temp.append(ret)  # append the simulated portfolio return to a temporary list to hold the simulations
            vol_temp.append(vol)  # append the simulated portfolio volatility to a temporary list to hold the simulations
            wts_temp.append(wts)  # append the simulated portfolio weight to a temporary list to hold the simulations
        rfr_list_esg.append([y[0], rfr])  # append the risk-free rate for that month period to its list
        # add the period, portfolio returns list and portfolio volatility list to the final returns and volatility list
        rdm_wt_port_ret_vol_esg.append([y[0], [ret_temp], [vol_temp]])
        # add the period and portfolio weights list to the final weights list
        rdm_wt_port_weights_esg.append([y[0], wts_temp])


    # Sharpe ratios and max Sharpe ratios
    # Calculate the sharpe ratios for each simulated portfolio and get the index of the highest sharpe ratio for each date
    rdm_wt_max_sharpe_indx = get_sharpe(test_ret_vol, rfr_list)
    rdm_wt_max_sharpe_indx_esg = get_sharpe(rdm_wt_port_ret_vol_esg, rfr_list)


    # Prepare the target
    # Prepare the targets i.e. Portfolio weights
    # Call function get_target to compute the weights of the max sharpe index for each date in the dataset
    train_target, train_target_with_dates = get_target(list(train_features_df.index), rdm_wt_max_sharpe_indx, test_port_weights)
    test_target, test_target_with_dates = get_target(list(test_features_df.index), rdm_wt_max_sharpe_indx, test_port_weights)
    train_target_esg, train_target_esg_with_dates = get_target(list(train_features_esg_df.index), rdm_wt_max_sharpe_indx_esg, rdm_wt_port_weights_esg)
    test_target_esg, test_target_esg_with_dates = get_target(list(test_features_esg_df.index), rdm_wt_max_sharpe_indx_esg, rdm_wt_port_weights_esg)

    # Format the target data.
    # Remove the duplicated data from the vstack method in the get_target function
    train_target = train_target[int(len(train_target)/2):]
    test_target = test_target[int(len(test_target)/2):]
    train_target_esg = train_target_esg[int(len(train_target_esg)/2):]
    test_target_esg = test_target_esg[int(len(test_target_esg)/2):]
    train_target_with_dates = train_target_with_dates[int(len(train_target_with_dates)/2):]
    test_target_with_dates = test_target_with_dates[int(len(test_target_with_dates)/2):]
    train_target_esg_with_dates = train_target_esg_with_dates[int(len(train_target_esg_with_dates)/2):]
    test_target_esg_with_dates = test_target_esg_with_dates[int(len(test_target_esg_with_dates)/2):]


    # Set the date as the index and compute the expected returns
    train_target_df = pd.DataFrame(train_target_with_dates, columns=['Date', 'Weights']).set_index('Date')
    test_target_df = pd.DataFrame(test_target_with_dates,  columns=['Date', 'Weights']).set_index('Date')
    train_target_esg_df = pd.DataFrame(train_target_esg_with_dates, columns=['Date', 'Weights']).set_index('Date')
    test_target_esg_df = pd.DataFrame(test_target_esg_with_dates,  columns=['Date', 'Weights']).set_index('Date')


    # Call the defined function random_forest to run the regression on both the non_ESG filtered and ESG_filtered datasets
    train_score, test_score, feat_importances, train_pred, test_pred = random_forest(train_features, train_target,
                                                                                     test_features, test_target,
                                                                                     besthyp['n_estimators'],
                                                                                     besthyp['max_depth'],
                                                                                     besthyp['max_features'],
                                                                                     besthyp['random_state'])

    esg_train_score, esg_test_score, esg_feat_importances, train_pred_esg, test_pred_esg = random_forest(
                                                                                    train_features_esg,
                                                                                    train_target_esg, test_features_esg,
                                                                                    test_target_esg,
                                                                                    esg_besthyp['n_estimators'],
                                                                                    esg_besthyp['max_depth'],
                                                                                    esg_besthyp['max_features'],
                                                                                    esg_besthyp['random_state'])

    # Get the test set portion of the monthly returns dataset
    monthly_returns_test_data = sp500_monthlyreturns[sp500_monthlyreturns.index.isin(test_target_df.index)]
    monthly_returns_test_data_esg = sp500esg_monthlyreturns[sp500esg_monthlyreturns.index.isin(test_target_esg_df.index)]

    # Get the S&P500 index monthly returns data
    monthly_returns_sp500_index = sp500_index_monthly_returns[sp500_index_monthly_returns.index.isin(
        test_target_df.index)]['SP500']

    # Multiply the weights predicted from the test data (test_pred)  with the monthly returns test data
    predicted_returns = np.sum(monthly_returns_test_data * test_pred, axis=1)
    predicted_returns_esg = np.sum(monthly_returns_test_data_esg * test_pred_esg, axis=1)

    # Format the index
    predicted_returns.index = predicted_returns.index.astype('<M8[ns]')
    predicted_returns_esg.index = predicted_returns_esg.index.astype('<M8[ns]')
    monthly_returns_sp500_index.index = monthly_returns_sp500_index.index.astype('<M8[ns]')


    # Benchmark predicted returns to view model performance

    # set both portfolios starting amounts to 10000 and calculate total returns over the period
    amount = 10000
    randomforest_total_returns = calculate_hypothetical_portfolio_returns(predicted_returns, amount, [amount])
    randomforest_total_returns_esg = calculate_hypothetical_portfolio_returns(predicted_returns_esg, amount, [amount])
    sp500index_total_returns = calculate_hypothetical_portfolio_returns(monthly_returns_sp500_index, amount, [amount])

    # Get the difference between the two total portfolio returns
    returns_diff = randomforest_total_returns - sp500index_total_returns
    returns_diff_esg = randomforest_total_returns_esg - sp500index_total_returns


esg_returns.append(randomforest_total_returns_esg)
nonesg_returns.append(randomforest_total_returns)

# Convert data to numpy arrays to be used in hypothesis test
esg_returns = np.array(esg_returns)
nonesg_returns = np.array(nonesg_returns)


# HYPOTHESIS TEST

# Define null hypothesis:
np.mean(esg_returns)-np.mean(nonesg_returns) == 0

# Calculate the difference between the observed means of the ESG-filtered and non-ESG-filtered data
mean_diffs_from_runs = np.mean(esg_returns) - np.mean(nonesg_returns)

# Call function perm_reps_mean_diff to compute the mean on 1000 permutation replicates of the returns data
perm_rep = perm_reps_mean_diff(esg_returns, nonesg_returns, size=1000)

# Compute p-value
p_value = np.sum(perm_rep >= mean_diffs_from_runs)/len(perm_rep)

# Interprete the results
if p_value > 0.05:
    print('p-value = ' + str(p_value) + ': The null hypothesis that the mean total portfolio returns from the non-ESG-filtered data '
    'is equal to the mean total returns of the ESG-filtered data is not rejected at the 5% significance level')

else:
    print('p-value = ' + str(p_value) + ': The null hypothesis that the mean total portfolio returns from the non-ESG-filtered data '
                   'is equal to the mean total returns of the ESG-filtered data is rejected at the 5% significance level')



