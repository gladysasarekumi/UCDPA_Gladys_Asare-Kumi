# Determining if an ESG first approach to portfolio construction significantly impacts on portfolio returns by
# comparing two portfolio construction methods. The first method uses the random forest machine learning method to
# preselect securities from the S&P500 index, filters by a predetermined ESG rating score and then uses
# mean-variance method to assign portfolio weights. The second uses an ESG first method to filter securities,
# utilizing the random forest prediction and mean-variance optimization.
# Accessing dictionary items: sp500_cov[pd.Period('2018-11', 'M')]

# Import the necessary modules
import numpy
import pandas as pd
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar
import seaborn as sns
import matplotlib.pyplot as plt
import talib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid
import itertools


# Functions to be used
def split_stratified_df(stratDf, colDate, colGrp, splitFrac, sortAsc1 = True, sortAsc2 = True):
    """Splits a stratified time series dataframe into 2 subsets based on the split percentage
    Args:
    stratDF (str): The dataframe to be split
    colDATE (str): The column label of the column with the dates for sorting
    colGRP (str): The column label of the column with the groups which need to be preserved
    splitFRAC (float): The fraction to be used for splitting
    sortASC1 (bool, optional): Default = True. Whether to sort colDATE in ascending order
    sortASC2 (bool, optional): Default = True. Whether to sort colGRP in ascending order
    Returns:
    DataFrame"""
    # Get column labels
    stratDf.sort_values([colDate, colGrp], ascending=[sortAsc1, sortAsc2])
    groups = list(stratDf[colGrp].unique())
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()

    for x in groups:
        temp_df = stratDf[stratDf[colGrp] == x]
        spltsize = int(splitFrac * temp_df.shape[0])
        temp1 = temp_df[:spltsize]
        temp2 = temp_df[spltsize:]
        df1 = df1.append(temp1)
        df2 = df2.append(temp2)
    df1 = df1
    df2 = df2
    return df1, df2


# Calculate covariances per each month and store the month as keys and covariances as values in a dictionary
def monthly_covariances_calc_daily(daily_returns_df, monthly_returns_df, dof=1):
    """Calculates the daily covariances for each month
        Args:
        daily_returns_df (series): The
        monthly_returns_df: The column labels of the columns whose rows are being compared
        dof (int, optional):
        Returns: int
        """
    cov_list = []
    for x in monthly_returns_df.index:
        indx = daily_returns_df.index
        mask = (indx.month == x.month) & (indx.year == x.year)
        # calculate daily covariances for each month using mask
        cov_list.append([x, daily_returns_df[mask].cov(ddof=dof)])
    return cov_list


# Calculate pct_changes to be used as features
# Calculate 10d_ClosePct, 1d_VolChangePct and 3d_VolChangePct
def pct_change_features(df, days, attribute):
    """Calculates the percentage changes in the dataframe
        Args:
        df (DataFrame): The dataframe containing the data to calculate percentage changes on
        days:
        attribute:
        Returns:
        """
    feature = df[attribute].pct_change(days)
    return feature


# 1. Calculate close price and Volume moving averages and rsi for 14, 30 and 200 time periods and add to features list
# ma14close, ma30close, ma50close, ma14volume, ma30volume, ma50volume, rsi14close, rsi30close, rsi50close, rsi14volume,
# rsi30volume, rsi50volume
def moving_average_feature(df, ma_days, ma_attribute):
    """Calculates the moving average for the DataFrame passed
            Args:
            df (DataFrame): The
            ma_days(int): The
            ma_attribute (series):
            Returns:
            """
    # Calculate SMA
    feature = df[str(ma_attribute)].transform(lambda n: talib.SMA(n.astype(
                float).values, timeperiod=ma_days))
    # Return calculated moving average
    return feature


# 1. Calculate close price and Volume moving averages and rsi for 14, 30 and 200 time periods and add to features list
# ma14close, ma30close, ma50close, ma14volume, ma30volume, ma50volume, rsi14close, rsi30close, rsi50close, rsi14volume,
# rsi30volume, rsi50volume
def relative_strength_index_feature(df, rsi_days, rsi_attribute):
    """Calculates the relative strength index (rsi) for the DataFrame passed
    Args:
    df (DataFrame): The
    rsi_days(int): The
    rsi_attribute (series):
    Returns:
                """
    # Calculate RSI
    feature = df[str(rsi_attribute)].transform(lambda n: talib.RSI(n.astype(
                float).values, timeperiod=rsi_days))
    # Return the calculated feature
    return feature


# Identify target (for each date (i.e. key) in the max_sharpe index list, return the max sharpe index)
def get_target(dates_list, max_sharpe_idx_list, portfolio_weights_list):
    """
                Args:
                df (DataFrame): The
                ma_days(int): The
                ma_attribute (series):
                Returns:
                """
    rows = len(dates_list)
    cols = len(portfolio_weights_list[0][1][0])
    target = np.empty(shape=(rows, 2))
    targarray = np.empty(shape=(rows, cols))
    for i in dates_list:
        for j in range(len(max_sharpe_idx_list)):
            if max_sharpe_idx_list[j][0] == i:
                max_sharpe_index = max_sharpe_idx_list[j][1]
                break
        for k in range(len(portfolio_weights_list)):
            if portfolio_weights_list[k][0] == i:
                max_weights = np.array(portfolio_weights_list[k][1][max_sharpe_index])
                targarray = np.vstack((targarray, [max_weights]))
                temp = np.array([[i], [max_weights]]).T
                target = np.vstack((target, temp))
                break
    return targarray,  target


# Multiply the amounts in our predicted returns model by looping by 1 + r in order to apply the returns to the amount
def calculate_hypothetical_portfolio_returns(data, amount, initial_amount):
    """Calculates the total returns of
    Args:
    data (Series): The
    amount(float64): The
    initial_amount (float64):
    Returns: float64
    """
    for x in data:
        amount = amount * (x+1)
        initial_amount.append(amount)
    # Calculate the total returns for the portfolio over the period by subtracting the starting balance from the
    # ending balance and dividing by the starting balance
    total_returns = initial_amount[-1] - initial_amount[0] / initial_amount[0]
    return total_returns


def random_forest(X_train, y_train, X_test, y_test, n=200, random_state=42, maxf=3, maxd=3):
    # Fit the model and get the scores on train and test
    random_forest = RandomForestRegressor(n_estimators=n, random_state=random_state, max_features=maxf, max_depth=maxd)
    random_forest.fit(X_train, y_train)
    train_score = random_forest.score(X_train, y_train)
    test_score = random_forest.score(X_test, y_test)
    feat_imp = random_forest.feature_importances_
    train_prediction = random_forest.predict(X_train)
    test_prediction = random_forest.predict(X_test)
    return train_score, test_score, feat_imp, train_prediction, test_prediction


def tune_hyperparameters(X_train, y_train, X_test, y_test):
    # Run a new random forest regressor on the data to be able to set and access the params attribute
    rf = RandomForestRegressor(n_estimators=300, random_state=42)
    rf.fit(X_train, y_train)
    # Dictionary of hyperparameters
    hypdict = {'n_estimators': [200, 300, 400, 500], 'max_depth': [2, 3, 4, 5],
           'max_features': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 'random_state': [42]}
    hypscores = []  # To hold calculated scores of the various combinations of hyperparameters

    # Loop through the dictionary, calculate hyperparameters and store
    for f in ParameterGrid(hypdict):
        rf.set_params(**f)
        rf.fit(X_train, y_train)
        hypscores.append((rf.score(X_test, y_test)))

    # Determine the best set of hyperparameters to use
    besthyp = np.argmax(hypscores)

    # return the best set of hyperparameters to use
    return hypscores[besthyp], ParameterGrid(hypdict)[besthyp]



# A: Import the data to be used

# Import S&P500 index historical data
sp500_index = pd.DataFrame(pd.read_csv("C:/Users/Gladys/Desktop/Data Science/Project/Data/SP500.csv", parse_dates=['DATE']))

# Import S&P500 constituents historical price data
sp500_df = pd.DataFrame(pd.read_csv("C:/Users/Gladys/Desktop/Data Science/Project/Data/all_stocks_5yr.csv"))

# Import Effective Federal Funds Rate data (this will be used as the risk free rate)
ffr_df = pd.DataFrame(pd.read_csv("C:/Users/Gladys/Desktop/Data Science/Project/Data/DFF.csv"))

# Import ESG Rating data file
esg_rat = pd.DataFrame(pd.read_csv("C:/Users/Gladys/Desktop/Data Science/Project/Data/ESGOverallRating.csv"))


# B: Data Preparation

# B1: Prepare ESG data:
esg_rat = esg_rat[['ISSUER_TICKER', 'IVA_COMPANY_RATING', 'IVA_RATING_DATE']]
esg_rat.index = pd.to_datetime(esg_rat['IVA_RATING_DATE'])
esg_rat.sort_index(inplace=True)
esg_rat.drop_duplicates(subset=['ISSUER_TICKER'], keep='last', inplace=True)

# B2: Prepare S&P500 index data
# Check for missing values
sp500_index.isnull().sum()  # Count the total number of NaNs...Appears to have no missing values

# To be sure there are no non_numeric values, coerce all values in the SP500 column to numerical. This will set
# any occurrence of non-numeric data to NaN
sp500_index['SP500'] = pd.to_numeric(sp500_index['SP500'], errors='coerce')

# Check again for missing values
sp500_index.isnull().sum()  # Now it returns a count of missing data

# Get the rows with missing data
missing_data_rows = sp500_index[sp500_index.isnull().any(axis=1)]

# Scanning through bad_data_rows, I suspect the rows are US bank holidays so I will check to the US Federal Holiday
# Calendar to be sure
caldr = USFederalHolidayCalendar()  # initialize the calendar variable and set it to the US Federal Holiday Calendar

# Import US bank holidays between the min and max dates in the S&P500 index data
hols = caldr.holidays(start=min(sp500_index['DATE']).date(), end=max(sp500_index['DATE']).date())

# Check which of the missing data row dates are not a US bank holiday
print(len(missing_data_rows[missing_data_rows.SP500.isin(hols)]))  # Returns 0 => all missing data are US bank holidays

# Remove the missing rows as we can be sure they are all bank holidays when there was no stock market trading
sp500_index = sp500_index.dropna()

# Set date column as the index
sp500_index.set_index('DATE',  inplace=True)

# Resample S&P500 index data monthly
sp500_index = sp500_index.resample('BMS').first()

# Convert dates to show format mm-yyyy
sp500_index.index = pd.to_datetime(sp500_index.index, format="%Y%m").to_period('M')

# Calculate returns
sp500_index_monthly_returns = sp500_index.pct_change().dropna()


# B4: Prepare S&P500 constituents historical price data
# Filter S&P500 constituents historical price data to only keep tickers that have been a constituent of the index for
# the whole period of analysis (Feb 2013 to Dec 2018)
# First, get list of all the unique dates in the datasets and unique list of tickers
fulldates = sp500_df['date'].unique()  # get all unique dates within the data
fulltickers = sp500_df['Name'].unique()  # get all unique tickers within the data

# Second, get all the possible ticker and date combinations if all the unique tickers in the dataset were constituents
# of the index throughout the period of analyis
fullDatesTickers = []  # initialize a list variable to hold the date and ticker combinations
for i in itertools.product(fulldates, fulltickers):
  fullDatesTickers.append("".join(map(str, i)))  # add the date and ticker combination to the list

# Third, get a list of the dates and ticker combinations that are actually present in the data
availableDatesTickers = list(sp500_df['date'] + sp500_df['Name'])

# Fourth, find the difference between the two list and create a list of the differences
diffs = list(set(fullDatesTickers) - set(availableDatesTickers))  # compare the two lists to find the differences
tickersToRemove = list(set([x[10:] for x in diffs]))  # create a list of all the unavailable combinations in the data

# Finally, filter the data to exclude the differences. This is the final data that the analysis will be perfomed on
sp500_df = sp500_df[~sp500_df.Name.isin(tickersToRemove)]


# Prepare ESG filtered data
# ESG Data
# Merge esg ratings data with S&P price data
sp500esg_df = sp500_df.merge(esg_rat, how='inner', left_on='Name', right_on='ISSUER_TICKER')

# Get the list of possible esg ratings
ratings = sorted(list(sp500esg_df['IVA_COMPANY_RATING'].unique()))

# To appy a ESG-first approach, a predetermined ESG rating value is set, then the available securities are filtered to
# include on those with that ESG rating or higher. That becomes the set of securities that our portfolio will be
# constructed from. This can then be fed into the random weights selection algorithm.
rating_limit = 'AAA'
sp500esg_df = sp500esg_df[sp500esg_df['IVA_COMPANY_RATING'] >= rating_limit]
sp500esg_df.reset_index(inplace=True, drop=True) # Reset the index to RangeIndex

# Pivot the sp500 data so tickers are in columns for ease of analysis
pivoted_sp500_df = pd.pivot(sp500_df, index='date', columns='Name', values='close')
pivoted_sp500esg_df = pd.pivot(sp500esg_df, index='date', columns='Name', values='close')

# Convert the index to a datetime dtype
pivoted_sp500_df.index = pd.to_datetime(pivoted_sp500_df.index)
pivoted_sp500esg_df.index = pd.to_datetime(pivoted_sp500esg_df.index)

# Sort the index
pivoted_sp500_df = pivoted_sp500_df.sort_index()
pivoted_sp500esg_df = pivoted_sp500esg_df.sort_index()

# Assume that the portfolio is rebalanced monthly during which time securities may be sold or bought
# Resample SP500 price data and risk free rate monthly
sp500monthly_df = pivoted_sp500_df.resample('BMS').first()
sp500esgmonthly_df = pivoted_sp500esg_df.resample('BMS').first()

# Convert dates to show format mm-yyyy
sp500monthly_df.index = pd.to_datetime(sp500monthly_df.index, format="%Y%m").to_period('M')
sp500esgmonthly_df.index = pd.to_datetime(sp500esgmonthly_df.index, format="%Y%m").to_period('M')

# Calculate daily and monthly returns
sp500_dailyreturns = pivoted_sp500_df.pct_change().dropna()  # Daily returns
sp500esg_dailyreturns = pivoted_sp500esg_df.pct_change().dropna()  # ESG Daily returns
sp500_monthlyreturns = sp500monthly_df.pct_change().dropna()  # Monthly returns
sp500esg_monthlyreturns = sp500esgmonthly_df.pct_change().dropna()  # ESG Monthly returns


# B5: Prepare the Effective federal funds rate data

# First, create a month/year column to use for grouping the data to average the rates monthly
ffr_df['Month&Year'] = pd.to_datetime(ffr_df['DATE']).dt.strftime('%B-%Y')  # Create a month/year column

# Calculate the average rate for each month from the daily rates
temp_ffr = ffr_df.groupby(pd.to_datetime(ffr_df['DATE']).dt.strftime('%B-%Y')).agg('mean')

# Join the calculated monthly average rate with the original daily data
ffr_df = ffr_df.join(temp_ffr, on='Month&Year', how='inner', rsuffix='Avg')

# Set the DATE column as the index
ffr_df.set_index('DATE',  inplace=True)

# Change the index to a datetime dtype
ffr_df.index = pd.to_datetime(ffr_df.index)

# Resample the data to get monthly data
ffr_df_mth_avg = ffr_df.resample('BMS').first()

# Convert the date index to show format mm-yyyy
ffr_df_mth_avg.index = pd.to_datetime(ffr_df_mth_avg.index, format="%Y%m").to_period('M')
ffr_df_mth_avg = ffr_df_mth_avg['DFFAvg']  # Rename the Average monthly rate column



# C: Portfolio Analysis

# # Calculate covariances per each month and store the month as keys and covariances as values in a dictionary
sp500_cov = monthly_covariances_calc_daily(sp500_dailyreturns, sp500_monthlyreturns, dof=0)
sp500esg_cov = monthly_covariances_calc_daily(sp500esg_dailyreturns, sp500esg_monthlyreturns, dof=0)

# Calculate portfolio returns and volatility for each date for the none-ESG filtered data
# Calculating portfolio returns and volatility on 10000 simulations of portfolios using random weights assignment
rdm_wt_port_ret_vol = []  # initialize empty dictionary to hold the portfolio returns and volatility
rdm_wt_port_weights = []   # initialize empty dictionary to hold the portfolio weights
rfr_list = []  # initialize empty dictionary to hold the risk-free rate
x = len(sp500_monthlyreturns.columns)  # get number of stocks in monthly returns dataframe
# get portfolio performances at each month
for y in sp500_cov:
    covariance = y[1]
    ret_temp, vol_temp, wts_temp = [], [], []
    rfr = ffr_df_mth_avg[ffr_df_mth_avg.index.strftime('%Y-%m') == str(y[0])][0]
    for portfolio in range(10):  # run 10 simulations of the portfolio
        wts = np.random.random(x)  # generate random weights for each security in the portfolio
        wts = wts/np.sum(wts)  # normalize so all weights sum to 1
        returns = sp500_monthlyreturns[sp500_monthlyreturns.index.strftime('%Y-%m')== str(y[0])].values
        returns = returns.reshape(wts.shape)
        ret = np.dot(wts, returns) # calculate the return of the portfolio
        # calculate the monthly volatility of the portfolio
        vol = np.sqrt(np.dot(wts.T, np.dot(covariance, wts)))*np.sqrt(21) # multiplied by sqrt(21) to convert fro daily to monthly
        ret_temp.append(ret)
        vol_temp.append(vol)
        wts_temp.append(wts)
    rfr_list.append([y[0], rfr])
    rdm_wt_port_ret_vol.append([y[0], [ret_temp], [vol_temp]])  # add portfolio returns and volatility to the dictionary for this date
    rdm_wt_port_weights.append([y[0], wts_temp])  # add the portfolio weight to the dictionary


# Do the same for the ESG-filtered Data
# Calculating portfolio returns and volatility using random weights assignment
rdm_wt_port_ret_vol_esg = []  # initialize empty dictionary to hold the portfolio returns and volatility
rdm_wt_port_weights_esg = []   # initialize empty dictionary to hold the portfolio weights
rfr_list_esg = []  # initialize empty dictionary to hold the risk-free rate
x = len(sp500esg_monthlyreturns.columns)  # get number of stocks in monthly returns dataframe
# get portfolio performances at each month
for y in sp500esg_cov:
    covariance = y[1]
    ret_temp, vol_temp, wts_temp = [], [], []
    rfr = ffr_df_mth_avg[ffr_df_mth_avg.index.strftime('%Y-%m') == str(y[0])][0]
    for portfolio in range(10):  # run 10 simulations of the portfolio
        wts = np.random.random(x)  # generate random weights for each security in the portfolio
        wts = wts/np.sum(wts)  # normalize so all weights sum to 1
        returns = sp500esg_monthlyreturns[sp500esg_monthlyreturns.index.strftime('%Y-%m')== str(y[0])].values
        returns = returns.reshape(wts.shape)
        ret = np.dot(wts, returns) # calculate the return of the portfolio
        # calculate the monthly volatility of the portfolio
        vol = np.sqrt(np.dot(wts.T, np.dot(covariance, wts)))*np.sqrt(21) # multiplied by sqrt(21) to convert fro daily to monthly
        ret_temp.append(ret)
        vol_temp.append(vol)
        wts_temp.append(wts)
    rfr_list_esg.append([y[0], rfr])
    rdm_wt_port_ret_vol_esg.append([y[0], [ret_temp], [vol_temp]])  # add portfolio returns and volatility to the dictionary for this date
    rdm_wt_port_weights_esg.append([y[0], wts_temp])  # add the portfolio weight to the dictionary


# Calculate the sharpe ratios for each simulated portfolio and get the index of the highest sharpe ratio for each date
# # Compute sharpe ratio to determine the best portfolio to select
# rdm_wt_sharpe, rdm_wt_max_sharpe_indx = get_sharpe(rdm_wt_port_returns,rdm_wt_port_volatility,rfr_list)
rdm_wt_sharpe = []  # Initialize empty dictionary to hold the calculated sharpe ratios
rdm_wt_max_sharpe_indx = []  # Initialize empty dictionary to hold the best sharpe index for each date
rdm_wt_port_ret_vol = sorted(rdm_wt_port_ret_vol)
counter = 0
for y in rdm_wt_port_ret_vol:
    temp_sharpe = []
    date = y[0]
    temp_ret = y[1]
    temp_vol = y[2]
    rfr = rfr_list[counter][1]
    for x in range(len(temp_ret)):
        ret = temp_ret[0]
        vol = temp_vol[0]
        # exclude 0 volatilities as they cannot be used to compute sharpe ratio. This means that the final list will
        # only contain dates for which the sharpe ratio can be computed. The length of the final max sharpe ratio
        # index could therefore be less than the total months within the period.
        if all(vol) == 0:  # exclude 0 volatilities
            break
        else:
            sharpe_val = (ret-rfr)/vol
            temp_sharpe.append(sharpe_val)
    if all(vol) == 0:
        continue
    else:
        rdm_wt_sharpe.append([date, temp_sharpe])
        rdm_wt_max_sharpe_indx.append([date, np.argmax(temp_sharpe)])
    counter +=1

rdm_wt_sharpe = sorted(rdm_wt_sharpe)
rdm_wt_max_sharpe_indx = sorted(rdm_wt_max_sharpe_indx)


# # Do the same for the ESG-filtered data.
# Compute sharp ratio to determine the best portfolio to select
# rdm_wt_sharpe, rdm_wt_max_sharpe_indx = get_sharpe(rdm_wt_port_returns,rdm_wt_port_volatility,rfr_list)
rdm_wt_sharpe_esg = []  # Initialize empty dictionary to hold the calculated sharpe ratios
rdm_wt_max_sharpe_indx_esg = []  # Initialize empty dictionary to hold the best sharpe index for each date
rdm_wt_port_ret_vol_esg = sorted(rdm_wt_port_ret_vol_esg)
counter = 0
for y in rdm_wt_port_ret_vol_esg:
    temp_sharpe = []
    date = y[0]
    temp_ret = y[1]
    temp_vol = y[2]
    rfr = rfr_list_esg[counter][1]
    for x in range(len(temp_ret)):
        ret = temp_ret[0]
        vol = temp_vol[0]
        # exclude 0 volatilities as they cannot be used to compute sharpe ratio. This means that the final list will
        # only contain dates for which the sharpe ratio can be computed. The length of the final max sharpe ratio
        # index could therefore be less than the total months within the period.
        if all(vol) == 0:  # exclude 0 volatilities
            break
        else:
            sharpe_val_esg = (ret-rfr)/vol
            temp_sharpe.append(sharpe_val_esg)
    if all(vol) == 0:
        continue
    else:
        rdm_wt_sharpe_esg.append([date, temp_sharpe])
        rdm_wt_max_sharpe_indx_esg.append([date, np.argmax(temp_sharpe)])
    counter += 1

rdm_wt_sharpe_esg = sorted(rdm_wt_sharpe_esg)
rdm_wt_max_sharpe_indx_esg = sorted(rdm_wt_max_sharpe_indx_esg)



# D: Machine Learning
# Use machine Learning to predict the best portfolio weights to assign for each date for comparison with the
# random weight assignments.The Random Forest Regressor model will be used.

# Identify features and targets

# Determine the list of Features
# List of Features:
# 1.	0 days lapse after governance score change => 0d_GovScoreChange
# 2.	1 days lapse after governance score change => 1d_GovScoreChange
# 3.	3 days lapse after governance score change => 3d_GovScoreChange
# 4.	10 day current price changes => 10d_ClosePct
# 5.	1 day current volume changes => 1d_VolChangePct
# 6.	3 day current volume changes => 1d_VolChangePct
# 7.	Previous day open price less than close price
# 8.



# A.3: Split the data into train and test sets. I will use 80% of the data as training data
# and the remaining 20% as test data
#filtered_sp500 = sp500_df[sp500_df['Name'].isin(random_tickers)]
train_set, test_set = split_stratified_df(sp500_df, 'date', 'Name', 0.8)
train_set_esg, test_set_esg = split_stratified_df(sp500esg_df, 'date', 'Name', 0.8)
feat_df_list = [train_set, train_set_esg, test_set, test_set_esg]

# Create a datetime object of the date column and set the index to this
for x, df in enumerate(feat_df_list):
    feat_df_list[x].index = pd.to_datetime(feat_df_list[x]['date'])
    feat_df_list[x].sort_index(inplace=True)


# Calculate train_set and test_set features
# Define the list of feature names
features = ['10d_ClosePct', '1d_VolChangePct', '5d_VolChangePct', '1d_OpenCloseSpread', 'ma14close', 'ma30close',
            'ma50close', 'ma14volume', 'ma30volume', 'ma50volume', 'rsi14close', 'rsi30close', 'rsi50close',
            'rsi14volume', 'rsi30volume', 'rsi50volume']

for df in feat_df_list:
    df['10d_ClosePct'] = pct_change_features(df.groupby('Name'), 10, 'close')
    df['1d_VolChangePct'] = pct_change_features(df.groupby('Name'), 1, 'volume')
    df['5d_VolChangePct'] = pct_change_features(df.groupby('Name'), 5, 'volume')
    for x in [14, 30, 50]:
        for y in ['close', 'volume']:
            # MA calculation
            df['ma' + str(x) + str(y)] = moving_average_feature(df.groupby('Name'), x, y)
            # RSI calculation
            df['rsi' + str(x) + str(y)] = relative_strength_index_feature(df.groupby('Name'), x, y)


# # Remove missing values from calculated feature columns
# for df_set in [train_set, train_set_esg, test_set, test_set_esg]:
#     df_set.dropna(inplace=True)

# Remove missing values from calculated feature columns and
# resample the data to get monthly data. Aggregate the tickers by  averaging for each date before resampling monthly

for x, df in enumerate(feat_df_list):
    feat_df_list[x].dropna(inplace=True)  # remove missing values
    grouped_df = feat_df_list[x].groupby(feat_df_list[x].index).mean()  # Aggregate all the different tickers for each day by getting average values for the month
    grouped_df = grouped_df.resample('BMS').first()  # Resample to get monthly data keeping only first business day of month
    grouped_df.index = pd.to_datetime(grouped_df.index, format="%Y%m").to_period('M')  # Format the index as yyyy-mm
    grouped_df = grouped_df.filter(regex=("[0-9]|10/\d+/d.*"))  # Use regex to filter for only the features columns
    grouped_df = grouped_df.shift(1).dropna()  # Shift the features down by 1 so that the previous month's features are aligned to the current month's portfolio
    feat_df_list[x] = grouped_df  # reassign back to the list of dataframes

# Get the features dataframes for correlation analyis
train_features_df, train_features_esg_df, test_features_df, test_features_esg_df = \
    feat_df_list[0], feat_df_list[1], feat_df_list[2], feat_df_list[3]

# Get the feature arrays for the random forest regression
train_features, train_features_esg, test_features, test_features_esg = \
    np.array(feat_df_list[0]), np.array(feat_df_list[1]), np.array(feat_df_list[2]), np.array(feat_df_list[3])


# Prepare the targets
# Call function get_target to compute the weights of the max sharpe index for each date in the dataset
train_target, train_target_with_dates = get_target(list(train_features_df.index), rdm_wt_max_sharpe_indx, rdm_wt_port_weights)
test_target, test_target_with_dates = get_target(list(test_features_df.index), rdm_wt_max_sharpe_indx, rdm_wt_port_weights)
train_target_esg, train_target_esg_with_dates = get_target(list(train_features_esg_df.index), rdm_wt_max_sharpe_indx_esg, rdm_wt_port_weights_esg)
test_target_esg, test_target_esg_with_dates = get_target(list(test_features_esg_df.index), rdm_wt_max_sharpe_indx_esg, rdm_wt_port_weights_esg)

# remove the duplicated data from the vstack method in the get_target function
train_target = train_target[int(len(train_target)/2):]
test_target = test_target[int(len(test_target)/2):]
train_target_esg = train_target_esg[int(len(train_target_esg)/2):]
test_target_esg = test_target_esg[int(len(test_target_esg)/2):]
train_target_with_dates = train_target_with_dates[int(len(train_target_with_dates)/2):]
test_target_with_dates = test_target_with_dates[int(len(test_target_with_dates)/2):]
train_target_esg_with_dates = train_target_esg_with_dates[int(len(train_target_esg_with_dates)/2):]
test_target_esg_with_dates = test_target_esg_with_dates[int(len(test_target_esg_with_dates)/2):]

#
train_target_df = pd.DataFrame(train_target_with_dates, columns=['Date', 'Weights']).set_index('Date')
test_target_df = pd.DataFrame(test_target_with_dates,  columns=['Date', 'Weights']).set_index('Date')
train_target_esg_df = pd.DataFrame(train_target_esg_with_dates, columns=['Date', 'Weights']).set_index('Date')
test_target_esg_df = pd.DataFrame(test_target_esg_with_dates,  columns=['Date', 'Weights']).set_index('Date')

# Get feature correlations
# Combine targets and features into one dataframe in order to calculate target correlations with features
train_target_features = pd.concat([train_target_df, train_features_df], axis=1, join="inner")
test_target_features = pd.concat([test_target_df, test_features_df], axis=1, join="inner")
train_target_features_esg = pd.concat([train_target_esg_df, train_features_esg_df], axis=1, join="inner")
test_target_features_esg = pd.concat([test_target_esg_df, test_features_esg_df], axis=1, join="inner")


# Calculate the correlation matrix of target and features
train_corr = train_target_features.corr()
test_corr = test_target_features.corr()
train_corr_esg = train_target_features_esg.corr()
test_corr_esg = test_target_features_esg.corr()

# Plot heatmap of the correlation matrix of the non-ESG data to visualize the correlation between target and features
sns.heatmap(train_corr, annot=True, annot_kws={"size": 6})
plt.yticks(rotation=0, size=12)
plt.xticks(rotation=90, size=12)  # fix ticklabel directions and size
plt.tight_layout()
plt.show()


# Plot heatmap of the correlation matrix for ESG data features to visualize the correlation between target and features
sns.heatmap(train_corr_esg, annot=True, annot_kws={"size": 6})
plt.yticks(rotation=0, size=12)
plt.xticks(rotation=90, size=12)  # fix ticklabel directions and size
plt.tight_layout()
plt.show()



# RANDOM FOREST REGRESSION

# Run the random forest regression on both ESG and non-ESG filtered data and return the train and test scores,
# feature importances and predictions. The hyperparameters were chosen arbitrarily.
train_score, test_score, feat_importances, train_pred, test_pred = random_forest(train_features, train_target,
                                                                test_features, test_target, 300, 42, 3, 2)
esg_train_score, esg_test_score, esg_feat_importances, train_pred_esg, test_pred_esg = random_forest(train_features_esg,
                                                train_target_esg, test_features_esg, test_target_esg, 300, 42, 3, 2)

print('Before tuning the hyperparameters, the Non-ESG-filtered data scored (train:' + str(train_score) +
      ' and test:' + str(test_score))
print('Before tuning the hyperparameters, the ESG-filtered data scored (train:' + str(esg_train_score) +
      ' and test:' + str(esg_test_score))


# HYPERPARAMETER TUNING

# Tune the hyperparameters for both datasets
hypscore, besthyp = tune_hyperparameters(train_features, train_target, test_features, test_target)
esg_hypscore, esg_besthyp = tune_hyperparameters(train_features_esg, train_target_esg, test_features_esg, test_target_esg)


# Re-run the random_forest model on the ESG and non-ESG data again using the best calculated hyperparameters
train_score, test_score, feat_importances, train_pred, test_pred = random_forest(train_features, train_target,
                                                                    test_features, test_target, besthyp['n_estimators'],
                                                                    besthyp['max_depth'], besthyp['max_features'],
                                                                    besthyp['random_state'])

esg_train_score, esg_test_score, esg_feat_importances, train_pred_esg, test_pred_esg = random_forest(train_features_esg,
                                                                                    train_target_esg, test_features_esg,
                                                                                    test_target_esg,
                                                                                    esg_besthyp['n_estimators'],
                                                                                    esg_besthyp['max_depth'],
                                                                                    esg_besthyp['max_features'],
                                                                                    esg_besthyp['random_state'])

print('After tuning the hyperparameters, the Non-ESG-filtered data scored (train:' + str(train_score) +
      ' and test:' + str(test_score))
print('After tuning the hyperparameters, the ESG-filtered data scored (train:' + str(esg_train_score) +
      ' and test:' + str(esg_test_score))


# FEATURE IMPORTANCES
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
plt.show()  # show the plot


# PREDICTIONS

# Although the test scores are low, check model performance by comparing the total portfolio returns
# with the returns of the S&P 500 index data. If the model predicted weights portfolio beat the S&P500 index returns,
# then the model is quite good

# Get the test set portion of the monthly returns dataset
monthly_returns_test_data = sp500_monthlyreturns[sp500_monthlyreturns.index.isin(test_target_df.index)]
monthly_returns_test_data_esg = sp500esg_monthlyreturns[sp500esg_monthlyreturns.index.isin(test_target_esg_df.index)]

# Get the S&P500 index monthly returns data
monthly_returns_sp500_index = sp500_index_monthly_returns[sp500_index_monthly_returns.index.isin(
    test_target_df.index)]['SP500']

# Multiply the weights predicted from the test data (test_pred)  with the monthly returns test data
predicted_returns = np.sum(monthly_returns_test_data * test_pred, axis=1)
predicted_returns_esg = np.sum(monthly_returns_test_data_esg * test_pred_esg, axis=1)

# Prepare the datasets for plotting
predicted_returns.index = predicted_returns.index.astype('<M8[ns]')
predicted_returns_esg.index = predicted_returns_esg.index.astype('<M8[ns]')
monthly_returns_sp500_index.index = monthly_returns_sp500_index.index.astype('<M8[ns]')

# Plot the S&P500 index returns and the random forest predicted returns for both the ESG-filtered and non-ESG filtered
# data to visualize
plt.plot(predicted_returns, label='Predicted Returns', marker='o')
plt.plot(predicted_returns_esg, label='Predicted Returns ESG', marker='s')
plt.plot(monthly_returns_sp500_index, label='S&P500 Index Returns', marker='x')
plt.xlabel('Month')
plt.ylabel('Returns')
plt.xticks(rotation=90, size=12)  # fix ticklabel directions and size
plt.tight_layout()
plt.legend()
plt.show()


# From the plot, there are are periods when our predictions perform better than the S&P500 index and vice versa.
# There is little difference between the performance of the ESG-filtered model vs the non-ESG filtered model
# To determine whether our model predicts better overall over the period,
# I will use a hypothetical situation where I have a sum of EUR10000 to invest in each portfolio and compute total
# returns for each over the period


# set both portfolios starting amounts to 10000 and calculate total returns over the period
amount = 10000
randomforest_total_returns = calculate_hypothetical_portfolio_returns(predicted_returns, amount, [amount])
randomforest_total_returns_esg = calculate_hypothetical_portfolio_returns(predicted_returns_esg, amount, [amount])
sp500index_total_returns = calculate_hypothetical_portfolio_returns(monthly_returns_sp500_index, amount, [amount])

# Get the difference between the two total portfolio returns
returns_diff = randomforest_total_returns - sp500index_total_returns
returns_diff_esg = randomforest_total_returns_esg - sp500index_total_returns

print('The non-ESG-first hypothetical portfolio beats the S&P500 index returns by EUR', returns_diff)
print('This ESG-First hypothetical portfolio beats the S&P500 index returns by EUR', returns_diff_esg)




