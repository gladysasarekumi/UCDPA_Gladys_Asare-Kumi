# TOPIC: Does an ESG-First approach to optimized portfolio construction adversely impact portfolio returns?

# DESCRIPTION: Determining if an ESG first approach to portfolio construction significantly impacts on portfolio returns by
# comparing two hypothetical equity portfolio construction methods. The first method uses equities from the S&P500 index
# to build the portfolio without any consideration given to the security issuer's ESG rating.
# The second method filters the equities of the S&P500 index by a minimum ESG rating of AAA, which depicts an ESG
# industry leader, prior to portfolio construction. Random weights are assigned to the securities in each hypothetical
# equity portfolio by running 10 simulations of random weights for each date and for each portfolio. For each date,
# the weights of the simulated portfolio with the max Sharpe ratio is extracted and used as the target for machine
# learning optimization of portfolio weights. The RandomForestRegressor from scikit-learn is utilized to create a
# supervised learning model to predict the optimal portfolio weights that will maximize returns. The machine-learning
# predicted-returns from the two hypothetical portfolios are compared with each other and also compared with the return
# of the S&P 500 index over the same period to determine the performance of both hypothetical portfolios.


# Import the necessary modules
import pandas as pd
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar
import talib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid
import itertools


# Define Functions to be used:

def split_stratified_df(stratDf, colDate, colGrp, splitFrac, sortAsc1 = True, sortAsc2 = True):
    """Splits a stratified time series dataframe into 2 time range subsets based on the split percentage
    Args:
    stratDF (str): The dataframe to be split
    colDATE (str): The column label of the column with the dates for sorting
    colGRP (str): The column label of the column with the groups which need to be preserved
    splitFRAC (float): The fraction to be used for splitting
    sortASC1 (bool, optional): Default = True. Whether to sort colDATE in ascending order
    sortASC2 (bool, optional): Default = True. Whether to sort colGRP in ascending order
    Returns: DataFrame, DataFrame
    """
    stratDf.sort_values([colDate, colGrp], ascending=[sortAsc1, sortAsc2])  # sort the data
    groups = list(stratDf[colGrp].unique())  # Get group labels of the column to group the data by
    df1 = pd.DataFrame()  # initialize empty dataframe to hold the first subset
    df2 = pd.DataFrame()  # initialize empty dataframe to hold the second subset

    for x in groups:  # iterate through the different groups and split the data at each group level
        temp_df = stratDf[stratDf[colGrp] == x]
        spltsize = int(splitFrac * temp_df.shape[0])
        temp1 = temp_df[:spltsize]
        temp2 = temp_df[spltsize:]
        df1 = df1.append(temp1)
        df2 = df2.append(temp2)
    df1 = df1
    df2 = df2
    return df1, df2


def monthly_covariances_calc_daily(daily_returns_df, monthly_returns_df, dof=1):
    """Calculates daily covariances on a monthly basis and returns a list containing the pairwise covariances for each
        month
        Args:
        daily_returns_df (series): A DataFrame of daily asset returns
        monthly_returns_df (DataFrame): A Dataframe of monthly asset returns
        dof (int, optional): the degrees of freedom used in the covariance calculation
        Returns: list
        """
    cov_list = []  # initialize empty list to hold the pairwise covariances for each month
    excluded_months = []
    # For each month in the monthly returns dataframe, extract all the daily returns for that month from the daily
    # returns dataframe and use that daily data to calculate the monthly pairwise covariances
    for x in monthly_returns_df.index:
        indx = daily_returns_df.index
        mask = (indx.month == x.month) & (indx.year == x.year)
        # For months with only 1 observation, set the dof to 0 to compute the covariance. This is to prevent an error
        # stopping the script from running. The covariances are used to compute the portfolio volatilities and the
        # volatilities will compute as 0 for those months. I have included a condition to exclude 0 volatilities from
        # the volatility computations which will prevent these 0 matrix covariances from being used in the analysis.
        if len(daily_returns_df[mask]) == 1:
            dof = 0
        cov_list.append([x, daily_returns_df[mask].cov(ddof=dof)])  # calculate daily covariances for each month
    return cov_list, excluded_months


def pct_change_features(df, days, attribute):
    """Calculates the percentage changes in the dataframe
        Args:
        df (DataFrame): The dataframe containing the data to calculate percentage changes on
        days (int): The periods to shift for the pct_change calculation
        attribute(str): The column name of the dataframe column for which the pct_change is calculated
        Returns: Series
        """
    feature = df[attribute].pct_change(days)  #Calculate the pct_change() of the specified column for the period 'days'
    return feature


def moving_average_feature(df, ma_days, ma_attribute):
    """Calculates the moving average (MA) for specified column of the DataFrame passed
            Args:
            df (DataFrame): The DataFrame whose column the MA should be calculated on
            ma_days(int): The number of days for which the MA should be calculated
            ma_attribute (series): The column for which the MA should be calculated.
            Returns: Series
            """
    # Calculate SMA
    feature = df[str(ma_attribute)].transform(lambda n: talib.SMA(n.astype(
                float).values, timeperiod=ma_days))  #calculate the moving average with the specified parameters
    # Return series with the calculated MA
    return feature


def relative_strength_index_feature(df, rsi_days, rsi_attribute):
    """Calculates the relative strength index (RSI) for the DataFrame passed
    Args:
    df (DataFrame): The DataFrame whose column the RSI should be calculated on
    rsi_days(int): The number of days for which the RSI should be calculated
    rsi_attribute (series): The column for which the RSI should be calculated.
    Returns: Series
                """
    # Calculate RSI
    feature = df[str(rsi_attribute)].transform(lambda n: talib.RSI(n.astype(
                float).values, timeperiod=rsi_days))  #calculate the moving average with the specified parameters
    # Return series with the calculated RSI
    return feature


def get_target(dates_list, max_sharpe_idx_list, portfolio_weights_list):
    """Creates a list of all the available max Sharpe portfolio weights with their respective months for all the
    months in the dataframe
    Args:
    dates_list (DateIndex): The index of all the dates for which a target is required.
    max_sharpe_idx_list(list): A list of the index of the max Sharpe portfolio for each period to be used to index the
                                list of portfolio weights for each period
    portfolio_weights_list (list): A list of the portfolio weights from which the portfolio weights of the max Sharpe
                                    portfolio should be extracted from
    Returns:
    """
    rows = len(dates_list)  # Defines the maximum number of rows the return array will have
    cols = len(portfolio_weights_list[0][1][0])  #Defines the number of columns the return array will have
    target = np.empty(shape=(rows, 2))  # initialize an empty list to hold the targets for the regression
    targarray = np.empty(shape=(rows, cols))  # initialize an empty list to hold the targets for the correlations

    for i in dates_list:
        for j in range(len(max_sharpe_idx_list)):
            # Return the max Sharpe index if the feature period is found in the list of max Sharpe indexes
            if max_sharpe_idx_list[j][0] == i:
                max_sharpe_index = max_sharpe_idx_list[j][1]
                break  # exit the inner for loop once the max sharpe index for that date is found
        # For the date for which the max Sharpe index was returned, return the list of portfolio weights at the same
        # index as the max Sharpe index but from the list of portfolio weights for that date
        for k in range(len(portfolio_weights_list)):
            if portfolio_weights_list[k][0] == i:
                max_weights = np.array(portfolio_weights_list[k][1][max_sharpe_index])
                # Add the returned list of Max Sharpe portfolio weights to the 'targarray' array by stacking vertically
                targarray = np.vstack((targarray, [max_weights]))
                temp = np.array([[i], [max_weights]]).T  # create an array of the date and portfolio weights pair
                # Add the returned list of Max Sharpe portfolio weights to the 'target' array by stacking vertically
                target = np.vstack((target, temp))
                break   # exit this inner loop once the weights for the date i is returned
    return targarray,  target  # return the two arrays


# Multiply the amounts in our predicted returns model by looping by 1 + r in order to apply the returns to the amount
def calculate_hypothetical_portfolio_returns(data, initial_amount, amount):
    """Calculates the total returns of a portfolio of assets
    Args:
    data (Series): The expected monthly returns on the securities in the portfolio multiplied by the predicted weights
    initial_amount(float64): The initial amount invested in the portfolio
    amount (list): The starting amount available in the portfolio before the investment
    Returns: float64
    """
    for x in data:
        # multiply the initial amount invested by each predicted return and append to the amount list
        initial_amount = initial_amount * (x+1)
        amount.append(initial_amount)
    # Calculate the total returns for the portfolio over the period by subtracting the starting balance from the
    # ending balance and dividing by the starting balance
    total_returns = amount[-1] - amount[0] / amount[0]
    return total_returns  # return the portfolio total returns


def random_forest(X_train, y_train, X_test, y_test, n=200, random_state=42, maxf=3, maxd=3):
    """Defines and fits a random forest regressor with the specified parameters to the train and test data
    Args:
    X_train (array): The data for the feature variables (predictors) of the training data
    y_train (array): The data for the target variable (predicted) of the training data
    X_test (array): The data for the feature variables (predictors) of the test data
    y_test (array): The data for the target variable (predicted) of the test data
    n (int): n_estimators (see RandomForestRegressor docstrings)
    random_state (int):  (see RandomForestRegressor docstrings)
    maxf (int): max_features (see RandomForestRegressor docstrings)
    maxd (int): max_depth (see RandomForestRegressor docstrings)
    """
    # Fit the model and get the scores on train and test
    random_forest = RandomForestRegressor(n_estimators=n, random_state=random_state, max_features=maxf, max_depth=maxd)
    random_forest.fit(X_train, y_train)  # Fit the model
    train_score = random_forest.score(X_train, y_train)  # get the Rsquare score of the training data
    test_score = random_forest.score(X_test, y_test)  # get the Rsquare score of the test data
    feat_imp = random_forest.feature_importances_   # get the feature importances
    train_prediction = random_forest.predict(X_train)  # get the train data predicted values for the target
    test_prediction = random_forest.predict(X_test)   # get the test data predicted values for the target
    return train_score, test_score, feat_imp, train_prediction, test_prediction  # return the variables


def tune_hyperparameters(X_train, y_train, X_test, y_test):
    """Uses the sklearn ParameterGrid function to calculate optimal parameters for a defined random forest regressor
    Args:
    X_train (array): The data for the feature variables (predictors) of the training data
    y_train (array): The data for the target variable (predicted) of the training data
    X_test (array): The data for the feature variables (predictors) of the test data
    y_test (array): The data for the target variable (predicted) of the test data
    Returns: list, list
        """
    # Define and fit a new random forest regressor to the data to be able to set and access the params attribute
    rf = RandomForestRegressor(n_estimators=300, random_state=42)
    rf.fit(X_train, y_train)
    # Define a Dictionary of hyperparameters for ParameterGrid function to explore
    hypdict = {'n_estimators': [200, 300, 400, 500], 'max_depth': [2, 3, 4, 5],
           'max_features': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 'random_state': [42]}
    hypscores = []  # Initialize a list to hold calculated scores of the various combinations of hyperparameters

    # Loop through the dictionary, calculate hyperparameters and store in hypscores
    for f in ParameterGrid(hypdict):
        rf.set_params(**f)
        rf.fit(X_train, y_train)
        hypscores.append((rf.score(X_test, y_test)))

    # Determine the best set of hyperparameters to use
    besthyp = np.argmax(hypscores)

    # return the best set of hyperparameters to use and its Rsquared score
    return hypscores[besthyp], ParameterGrid(hypdict)[besthyp]


def remove_us_bank_holidays(df, colname, datecolname):
    """Removes dates which are US bank holidays from the specified column of a DataFrame of time series data which has
    missing values for those bank holidays
       Args:
       df (DataFrame): A dataframe with a column of dates to search on
       colname (str): The name of the column with the missing values
       datecolname (str): The name of the column with parsed dates
       Returns: DataFrame
           """
    # coerce all values in colname to numerical. This will set any occurrence of non-numeric data to NaN
    df[colname] = pd.to_numeric(df[colname], errors='coerce')
    missing_data_rows = df[df.isnull().any(axis=1)]  # Get the rows with missing data
    caldr = USFederalHolidayCalendar()  # Get US Federal Holiday calendar
    # Filter the calendar caldr to keep only those federal holidays within the data timeframe
    hols = caldr.holidays(start=min(df[datecolname]).date(), end=max(df[datecolname]).date())
    # Create a filter variable of all the NaNs with dates included the filtered calendar caldr
    subset = missing_data_rows[missing_data_rows[datecolname].isin(hols)][datecolname]
    # Subset our data to only include those rows with dates not included in the calendar caldr
    df = df[-df[datecolname].isin(subset)]
    return df  #return the filtered DataFrame


def get_sharpe(rets_vols, rfr_ls):
    """Calculates the sharpe ratio for each portfolio and the index of the max sharpe ratio from the list of portfolios
    Args:
    rets_vols (list): A list of lists of date periods and their respective returns and volatilities for each portfolio
    rfr_ls (list): A list of date periods and their respective risk free rates
    Returns: list
    """
    sharpe, max_sharpe_indx = [], []   #initialize empty list to hold the calculated sharpe ratio and max sharpe indexes
    rets_vols = sorted(rets_vols)  #sort the returns and volatility list
    for y in rets_vols:  # iterate over the list of list of simulated portfolio returns and list of portfolio volatilities
        date = y[0]  # get the date period from the rets_vols list
        ret = y[1][0]  # get the list of returns for that date period
        vol = y[2][0]  # get the list of volatilities for that date period
        rfr = [x[1] for x in rfr_ls if x[0] == date][0]  # get the risk free rate for that date period
        # exclude 0 volatilities as they cannot be used to compute sharpe ratio. This means that the final list will
        # only contain dates for which the sharpe ratio can be computed. The length of the final max sharpe ratio
        # index could therefore be less than the total months within the period.
        if all(vol) == 0:  # exclude 0 volatilities
            continue
        else:
            sharpe_val = (ret - rfr) / vol  # compute the Sharpe ratios for the list of portfolios for that date period
            sharpe.append([date, sharpe_val])  # append the computed Sharpe ratios to the sharpe list
            max_sharpe_indx.append([date, np.argmax(sharpe_val)])  # get the index of the max Sharpe ratio for that date
    max_sharpe_indx = sorted(max_sharpe_indx)  # sort the max Sharpe indexes

    return max_sharpe_indx  # return the sorted max Sharpe indexes


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



# A: READ IN THE DATA

# Import S&P500 index historical data into a DataFrame and parse the specified column of dates
sp500_index = pd.DataFrame(pd.read_csv("C:/Users/Gladys/Desktop/Data Science/Project/Data/SP500.csv", parse_dates=['DATE']))

# Import S&P500 constituents historical price data into a DataFrame and parse the specified column of dates
sp500_df = pd.DataFrame(pd.read_csv("C:/Users/Gladys/Desktop/Data Science/Project/Data/all_stocks_5yr.csv", parse_dates=['date']))

# Import Daily 3-Month Treasury Bill Rate data into a DataFrame and parse the specified column of dates
rfr_df = pd.DataFrame(pd.read_csv("C:/Users/Gladys/Desktop/Data Science/Project/Data/DTB3.csv", parse_dates=['DATE']))

# Import ESG Rating data file
esg_rat = pd.DataFrame(pd.read_csv("C:/Users/Gladys/Desktop/Data Science/Project/Data/ESGOverallRating.csv"))


# B: DATA PREPARATION

# B1: Prepare the imported ESG data:
esg_rat = esg_rat[['ISSUER_TICKER', 'IVA_COMPANY_RATING', 'IVA_RATING_DATE']]  # only keep these 3 columns
esg_rat.index = pd.to_datetime(esg_rat['IVA_RATING_DATE'])   # convert the column to a datetime and set as the index
esg_rat.sort_index(inplace=True)   # sort the index ascending
esg_rat.drop_duplicates(subset=['ISSUER_TICKER'], keep='last', inplace=True)  # keep only the most recent rating for each ticker


# B2: Prepare the imported S&P500 index data
# Check for missing values
print(sp500_index.isnull().sum())  # Count the total number of NaNs...Appears to have no missing values

# To be sure there are no non_numeric values, apply the remove_us_bank_holidays function to the dataframe. This function
# will convert all non-numeric values to NaN and then remove those NaNs that are a result of missing data due to a
# US bank holiday with US financial markets closed
sp500_index = remove_us_bank_holidays(sp500_index, 'SP500', 'DATE')

# Check for missing values again and if there are any, they are missing not because markets were closed so
# fill those missing values with the last observed value using fillna with ffill
if sp500_index['SP500'].isnull().sum() > 0:
    sp500_index.fillna(method='ffill')

# Set date column as the index
sp500_index.set_index('DATE',  inplace=True)

# Resample S&P500 index data monthly using the first business day of each month
sp500_index = sp500_index.resample('BMS').first()

# Convert dates to Period dtype to show format mm-yyyy
sp500_index.index = pd.to_datetime(sp500_index.index, format="%Y%m").to_period('M')

# Calculate the monthly index returns
sp500_index_monthly_returns = sp500_index.pct_change().dropna()


# B3: Prepare the imported S&P500 constituents historical price data
# Filter S&P500 constituents historical price data to only keep tickers that have been a constituent of the index for
# the whole period of analysis (Feb 2013 to Dec 2018)
# First, get list of all the unique dates in the datasets and unique list of tickers
fulldates = sp500_df['date'].astype(str).unique()  # get all unique dates within the data
fulltickers = sp500_df['Name'].unique()  # get all unique tickers within the data

# Second, get all the possible ticker and date combinations if all the unique tickers in the dataset were constituents
# of the index throughout the period of analyis
fullDatesTickers = []  # initialize a list variable to hold the date and ticker combinations
for i in itertools.product(fulldates, fulltickers):
  fullDatesTickers.append("".join(map(str, i)))  # add the date and ticker combination to the list

# Third, get a list of the dates and ticker combinations that are actually present in the data
availableDatesTickers = list(sp500_df['date'].astype(str) + sp500_df['Name'])

# Fourth, find the difference between the two list and create a list of the differences
diffs = list(set(fullDatesTickers) - set(availableDatesTickers))  # compare the two lists to find the differences
tickersToRemove = list(set([x[10:] for x in diffs]))  # create a list of all the unavailable combinations in the data

# Finally, filter the data to exclude the differences. This is the final data that the analysis will be perfomed on
sp500_df = sp500_df[~sp500_df.Name.isin(tickersToRemove)]


# B4:  Prepare imported ESG data
# Merge ESG ratings data with the S&P price data and return only the intersecting tickers
sp500esg_df = sp500_df.merge(esg_rat, how='inner', left_on='Name', right_on='ISSUER_TICKER')

# Get the list of unique esg ratings within the dataset
ratings = sorted(list(sp500esg_df['IVA_COMPANY_RATING'].unique()))

# To appy a ESG-first approach, a predetermined ESG rating value is set, then the available securities are filtered to
# include only those with that ESG rating or higher. That becomes the set of securities that the portfolio will be
# constructed from. The selected securities can then be fed into the random weights assignment algorithm.
rating_limit = 'AAA'  # define the minimum ESG rating to be applied
sp500esg_df = sp500esg_df[sp500esg_df['IVA_COMPANY_RATING'] >= rating_limit]  # filter by the predefined minimum rating
sp500esg_df.reset_index(inplace=True, drop=True)  # Reset the index to RangeIndex to allow the data to be pivoted


# B5: Further transformations required on the merged S&P500 and ESG data.
# Pivot the sp500 data so tickers are in columns for ease of analysis. The values to be used for analysis are the daily
# close prices.
pivoted_sp500_df = pd.pivot(sp500_df, index='date', columns='Name', values='close')  # perform on non-ESG filtered data
pivoted_sp500esg_df = pd.pivot(sp500esg_df, index='date', columns='Name', values='close')  # perform on ESG filtered data

# Convert the index to a datetime dtype
pivoted_sp500_df.index = pd.to_datetime(pivoted_sp500_df.index)
pivoted_sp500esg_df.index = pd.to_datetime(pivoted_sp500esg_df.index)

# Sort the index ascending
pivoted_sp500_df = pivoted_sp500_df.sort_index()
pivoted_sp500esg_df = pivoted_sp500esg_df.sort_index()

# Assume that the portfolio is rebalanced monthly during which time securities may be sold or bought
# Resample SP500 price data and risk free rate monthly using the first business day of the month
sp500monthly_df = pivoted_sp500_df.resample('BMS').first()
sp500esgmonthly_df = pivoted_sp500esg_df.resample('BMS').first()

# Convert dates to Monthly Period to show format mm-yyyy
sp500monthly_df.index = pd.to_datetime(sp500monthly_df.index, format="%Y%m").to_period('M')
sp500esgmonthly_df.index = pd.to_datetime(sp500esgmonthly_df.index, format="%Y%m").to_period('M')

# Calculate daily and monthly returns
sp500_dailyreturns = pivoted_sp500_df.pct_change().dropna()  # Daily returns
sp500esg_dailyreturns = pivoted_sp500esg_df.pct_change().dropna()  # ESG-filtered daily returns
sp500_monthlyreturns = sp500monthly_df.pct_change().dropna()  # Monthly returns
sp500esg_monthlyreturns = sp500esgmonthly_df.pct_change().dropna()  # ESG-filtered monthly returns


# B6: Prepare the Daily 3-month Treasury Bill rate data

# Check for missing values
print(rfr_df.isnull().sum())

# Remove bank holidays
rfr_df = remove_us_bank_holidays(rfr_df, 'DTB3', 'DATE')

# Forward fill remaining missing values
if rfr_df['DTB3'].isnull().sum() > 0:
    rfr_df.fillna(method='ffill')


# Create a month/year column to use for grouping the data in order to average the rates monthly
rfr_df['Month&Year'] = pd.to_datetime(rfr_df['DATE']).dt.strftime('%B-%Y')  # Create a month/year column

# Calculate the average rate for each month from the daily rates
temp_rfr = rfr_df.groupby(pd.to_datetime(rfr_df['DATE']).dt.strftime('%B-%Y')).agg('mean')

# Join the calculated monthly average rate with the original daily data
rfr_df = rfr_df.join(temp_rfr, on='Month&Year', how='inner', rsuffix='Avg')

# Set the DATE column as the index
rfr_df.set_index('DATE', inplace=True)

# Change the index to a datetime dtype
rfr_df.index = pd.to_datetime(rfr_df.index)

# Resample the data to get monthly data
rfr_df_mth_avg = rfr_df.resample('BMS').first()

# Convert the date index to show format mm-yyyy
rfr_df_mth_avg.index = pd.to_datetime(rfr_df_mth_avg.index, format="%Y%m").to_period('M')
rfr_df_mth_avg = rfr_df_mth_avg['DTB3Avg']  # Keep only the Average monthly rate column


# C: PORTFOLIO ANALYSIS

# C1: Covariances
# Calculate the pairwise security covariances per each month and store the month as keys and covariances as values
# in a dictionary. Call the defined function monthly_covariances_calc_daily to do this.
sp500_cov, exclusions = monthly_covariances_calc_daily(sp500_dailyreturns, sp500_monthlyreturns)
sp500esg_cov, exclusions_esg = monthly_covariances_calc_daily(sp500esg_dailyreturns, sp500esg_monthlyreturns)

# C2: Portfolio Returns, volatility and weights
# Calculate portfolio returns and volatility for each date for the non-ESG filtered data
rdm_wt_port_ret_vol = []  # initialize empty list to hold the portfolio returns and volatility
rdm_wt_port_weights = []   # initialize empty list  to hold the portfolio weights
rfr_list = []  # initialize empty list to hold the risk-free rate
x = len(sp500_monthlyreturns.columns)  # get number of stocks in the monthly returns dataframe
# get portfolio performances for each month
for y in sp500_cov:  # loop through the list of covariances
    covariance = y[1]
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
        vol = np.sqrt(np.dot(wts.T, np.dot(covariance, wts)))  # compute the portfolio volatility
        ret_temp.append(ret)  # append the simulated portfolio return to a temporary list to hold the simulations
        vol_temp.append(vol)  # append the simulated portfolio volatility to a temporary list to hold the simulations
        wts_temp.append(wts)  # append the simulated portfolio weight to a temporary list to hold the simulations
    rfr_list.append([y[0], rfr])  # append the risk-free rate for that month period to its list
    # add the period, portfolio returns list and portfolio volatility list to the final returns and volatility list
    rdm_wt_port_ret_vol.append([y[0], [ret_temp], [vol_temp]])
    # add the period and portfolio weights list to the final weights list
    rdm_wt_port_weights.append([y[0], wts_temp])


# Calculate portfolio returns and volatility for each date for the ESG filtered data
rdm_wt_port_ret_vol_esg = []  # initialize empty list to hold the portfolio returns and volatility
rdm_wt_port_weights_esg = []   # initialize empty list to hold the portfolio weights
rfr_list_esg = []  # initialize empty list to hold the risk-free rate
x = len(sp500esg_monthlyreturns.columns)  # get number of stocks in monthly returns dataframe
# get portfolio performances for each month
for y in sp500esg_cov:
    covariance = y[1]
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
        vol = np.sqrt(np.dot(wts.T, np.dot(covariance, wts)))  # compute the portfolio volatility
        ret_temp.append(ret)  # append the simulated portfolio return to a temporary list to hold the simulations
        vol_temp.append(vol)  # append the simulated portfolio volatility to a temporary list to hold the simulations
        wts_temp.append(wts)  # append the simulated portfolio weight to a temporary list to hold the simulations
    rfr_list_esg.append([y[0], rfr])  # append the risk-free rate for that month period to its list
    # add the period, portfolio returns list and portfolio volatility list to the final returns and volatility list
    rdm_wt_port_ret_vol_esg.append([y[0], [ret_temp], [vol_temp]])
    # add the period and portfolio weights list to the final weights list
    rdm_wt_port_weights_esg.append([y[0], wts_temp])


# C3: Sharpe ratios and max Sharpe ratios
# Calculate the sharpe ratios for each simulated portfolio and get the index of the highest sharpe ratio for each date
rdm_wt_max_sharpe_indx = get_sharpe(rdm_wt_port_ret_vol, rfr_list)
rdm_wt_max_sharpe_indx_esg = get_sharpe(rdm_wt_port_ret_vol_esg, rfr_list)


# D: SUPERVISED LEARNING

# Use machine Learning to predict the best portfolio weights to assign for each date period in order to optimize
# total porftolio returns.The Random Forest Regressor model will be used.

# D1: Get Test and Train sets
# Split the data into train and test sets. I will use 80% of the data as training data
# and the remaining 20% as test data
train_set, test_set = split_stratified_df(sp500_df, 'date', 'Name', 0.8)
train_set_esg, test_set_esg = split_stratified_df(sp500esg_df, 'date', 'Name', 0.8)
feat_df_list = [train_set, train_set_esg, test_set, test_set_esg]

# Create a datetime object of the date column and set the index to this
for x, df in enumerate(feat_df_list):
    feat_df_list[x].index = pd.to_datetime(feat_df_list[x]['date'])
    feat_df_list[x].sort_index(inplace=True)


# D2: Prepare Features
# Calculate train_set and test_set features

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


# Remove missing values from calculated feature columns and
# resample the data to get monthly data. Aggregate the tickers by  averaging for each date before resampling monthly

for x, df in enumerate(feat_df_list):
    feat_df_list[x].dropna(inplace=True)  # remove missing values
    grouped_df = feat_df_list[x].groupby(feat_df_list[x].index).mean()  # Aggregate all the different tickers for each day by getting average values for the month
    grouped_df = grouped_df.resample('BMS').first()  # Resample to get monthly data keeping only first business day of month
    grouped_df.index = pd.to_datetime(grouped_df.index, format="%Y%m").to_period('M')  # Format the index as yyyy-mm
    # # Remove all the months that were in the exclusions and exclusions_esg list. These are the months for which no
    # # volatility was calculated because the covariances could not be computed due to the number of observations being 1.
    # # This filtering is to ensure that the shape of the features and the targets match for the random forest regression.
    # if x == 0 or x == 2:
    #     grouped_df = grouped_df[~grouped_df.index.isin(exclusions)]
    # else:
    #     grouped_df = grouped_df[~grouped_df.index.isin(exclusions_esg)]
    grouped_df = grouped_df.filter(regex=("[0-9]|10/\d+/d.*"))  # Use regex to filter for only the features columns
    grouped_df = grouped_df.shift(1).dropna()  # Shift the features down by 1 so that the previous month's features are aligned to the current month's portfolio
    feat_df_list[x] = grouped_df  # reassign back to the list of dataframes


# Get the features dataframes for correlation analyis
train_features_df, train_features_esg_df, test_features_df, test_features_esg_df = \
    feat_df_list[0], feat_df_list[1], feat_df_list[2], feat_df_list[3]

# Get the feature arrays for the random forest regression
train_features, train_features_esg, test_features, test_features_esg = \
    np.array(feat_df_list[0]), np.array(feat_df_list[1]), np.array(feat_df_list[2]), np.array(feat_df_list[3])


# D3: Prepare the target
# Prepare the targets i.e. Portfolio weights
# Call function get_target to compute the weights of the max sharpe index for each date in the dataset
train_target, train_target_with_dates = get_target(list(train_features_df.index), rdm_wt_max_sharpe_indx, rdm_wt_port_weights)
test_target, test_target_with_dates = get_target(list(test_features_df.index), rdm_wt_max_sharpe_indx, rdm_wt_port_weights)
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


# D4: Target-Feature Correlations
# Prepare the target dataframes to be used for computing correlations
# Correlations will be computed and plotted only for the train data

# 1. set the date as the index and compute the expected returns
train_target_df = pd.DataFrame(train_target_with_dates, columns=['Date', 'Weights']).set_index('Date')
test_target_df = pd.DataFrame(test_target_with_dates,  columns=['Date', 'Weights']).set_index('Date')
train_target_esg_df = pd.DataFrame(train_target_esg_with_dates, columns=['Date', 'Weights']).set_index('Date')
test_target_esg_df = pd.DataFrame(test_target_esg_with_dates,  columns=['Date', 'Weights']).set_index('Date')

# 2. Filter the monthly returns data to include only the dates in the train data
monthly_returns_train_data = sp500_monthlyreturns[sp500_monthlyreturns.index.isin(train_target_df.index)]
monthly_returns_esg_train_data = sp500esg_monthlyreturns[sp500esg_monthlyreturns.index.isin(train_target_esg_df.index)]

# 3. Compute the expected returns by multiplying the monthly returns with the weights
train_target_df = np.sum(monthly_returns_train_data * train_target, axis=1)
train_target_esg_df = np.sum(monthly_returns_esg_train_data * train_target_esg, axis=1)

# 4. Combine targets and features into one dataframe in order to calculate target correlations with features
train_target_features = pd.concat([train_target_df, train_features_df], axis=1, join="inner")
train_target_features_esg = pd.concat([train_target_esg_df, train_features_esg_df], axis=1, join="inner")

# Rename the first column in the train data
train_target_features.rename(columns={list(train_target_features)[0]:'Expected Returns'}, inplace=True)
train_target_features_esg.rename(columns={list(train_target_features_esg)[0]:'Expected Returns'}, inplace=True)

# 6. Calculate the correlation matrix of target and features for the train data
train_corr = train_target_features.corr()
train_corr_esg = train_target_features_esg.corr()

# The heatmap for the train data correlations is plotted at the end of the script


# D5: RANDOM FOREST REGRESSION
# Run the random forest regression on both ESG and non-ESG filtered data and return the train and test scores,
# feature importances and predictions. The hyperparameters were chosen arbitrarily without tuning.

# Call the defined function random_forest to run the regression on both the non_ESG filtered and ESG_filtered datasets
train_score, test_score, feat_importances, train_pred, test_pred = random_forest(train_features, train_target,
                                                                test_features, test_target, 300, 42, 3, 2)
esg_train_score, esg_test_score, esg_feat_importances, train_pred_esg, test_pred_esg = random_forest(train_features_esg,
                                                train_target_esg, test_features_esg, test_target_esg, 300, 42, 3, 2)

# Print the test and train scores from both regressions
print('Before tuning the hyperparameters, the Non-ESG-filtered data scored (train:' + str(train_score) +
      ' and test:' + str(test_score))
print('Before tuning the hyperparameters, the ESG-filtered data scored (train:' + str(esg_train_score) +
      ' and test:' + str(esg_test_score))


# D6: HYPERPARAMETER TUNING

# Tune the hyperparameters for both datasets by passing the datasets to the defined function tune_hyperparameters
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

# Print the test and train scores of both models after hyperparameter tuning
print('After tuning the hyperparameters, the Non-ESG-filtered data scored (train:' + str(train_score) +
      ' and test:' + str(test_score) + ')')
print('After tuning the hyperparameters, the ESG-filtered data scored (train:' + str(esg_train_score) +
      ' and test:' + str(esg_test_score) + ')')
print('The Rsquared scores are' + str(hypscore) + ' and ' + str(esg_hypscore) + ' respectively.')

# D7: PREDICTIONS

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
# The plot comparing the predicted returns of the ESG-Filtered data and the non-ESG filtered data with the returns of the
# S&P500 index is at the end of the script under VISUALIZATIONS


# D8: Benchmark predicted returns to view model performance

# From the plot, there are are periods when the model predictions perform better than the S&P500 index and vice versa.
# There is little difference between the performance of the ESG-filtered model vs the non-ESG filtered model
# To determine whether our model predicts better overall over the period,
# I will use a hypothetical situation where I invest an amount of EUR10000 in each portfolio and compute total
# returns for each over the test period


# Set both portfolios starting amounts to 10000 and calculate total returns over the period
amount = 10000
randomforest_total_returns = calculate_hypothetical_portfolio_returns(predicted_returns, amount, [amount])
randomforest_total_returns_esg = calculate_hypothetical_portfolio_returns(predicted_returns_esg, amount, [amount])
sp500index_total_returns = calculate_hypothetical_portfolio_returns(monthly_returns_sp500_index, amount, [amount])

# Get the difference between the two total portfolio returns
returns_diff = randomforest_total_returns - sp500index_total_returns
returns_diff_esg = randomforest_total_returns_esg - sp500index_total_returns

print('The non-ESG-first hypothetical portfolio beats the S&P500 index returns by EUR', returns_diff)
print('This ESG-First hypothetical portfolio beats the S&P500 index returns by EUR', returns_diff_esg)



