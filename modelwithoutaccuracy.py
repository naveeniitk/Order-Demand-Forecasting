
import pandas as pd
import numpy as np
#from google.colab import drive
#drive.mount('/content/drive')
# G = pd.read_csv(r'C:\Users\Dell\Downloads\final.csv') # srilekha
G = pd.read_csv(r'E:\nk\Internship\piyush_fynd-demandforecasting-ebf9ac564e2a\piyush_fynd-demandforecasting-ebf9ac564e2a\New_full_data_27.csv')# naveen
import matplotlib.pyplot as plt
import datetime
# convert the datetime column to a datetime type and assign it back to the column
# df_dd.datetime = pd.to_datetime(df_dd.datetime)

G.columns

def fill_in_missing_dates(df, date_col_name = 'date',fill_val = 0.1 ,date_format='%Y-%m-%d'):
    df.set_index(date_col_name, drop=True, inplace=True)
    df.index = pd.to_datetime(df.index, format = date_format)
    idx = pd.date_range(df.index.min(), df.index.max())
    print('missing_dates are',idx.difference(df.index))
    df=df.reindex(idx,fill_value=fill_val)
    print('missing_dates after fill',idx.difference(df.index))
    df[date_col_name] = df.index
    df.reset_index(drop=True,inplace=True)
    return df

H = fill_in_missing_dates(G, 'Day of Order Date', 0.0)

# H.head()
l = ['Day of Order Date', 'AIR CARE', 'AIR CONDITIONER SERVICE', 'AUDIO', 'AUDIO ENHANCEMENT',
       'CNP IT', 'CNP MOBILITY', 'CONSUMABLES', 'COOKING APPLIANCE SERVICE',
       'COOKING APPLIANCES', 'DESKTOP', 'DISHWASHER SERVICE',
       'Delivery Charges', 'ENTERTAINMENT PROMOTION', 'ERECHARGE', 'FANS',
       'FOOD PRESERVATION', 'GAMING ACCESSORY', 'HARDWARE MOBILITY',
       'HIGH END TV', 'HOME APPLIANCES PROMOTION', 'HOME AUDIO SERVICE',
       'HOME CARE SERVICE', 'HOME VISUAL SERVICE', 'LAPTOP', 'LAPTOP SERVICE',
       'LAUNDRY & WASH CARE', 'LIFESTYLE IT', 'LIFESTYLE MOBILITY', 'LIGHTING',
       'MEDIAPLAYER', 'MUSICAL INSTRUMENTS (POWERED)',
       'Mobility (Jio Branded)', 'PERIPHERAL', 'PHOTOGRAPHY',
       'PHOTOGRAPHY WEARABLE', 'POWER', 'POWER ENTERTAINMENT', 'POWER HA',
       'PRINTER', 'Printer Services', 'REFRIGERATOR SERVICE',
       'SMALL DOMESTIC APPL', 'SMART HOME DEVICES', 'SOFTWARE', 'STORAGE',
       'TABLET', 'TABLET SERVICE', 'TECH ACCESSORIES IT',
       'TECH ACCESSORIES MOBILITY', 'TELECOM SERVICE', 'WASHER SERVICE',
       'WEARABLE DEVICE', 'WEARABLES SERVICE', 'WIRELESS PHONE',
       'WIRELESS PHONE SERVICE', 'BIG BAZAAR-PROD', 'DC / STORE CONSUMABLES',
       'DRYER SERVICE', 'GAMING SOFTWARE', 'HARDWARE HA']

T = H[l]
T.head()

G = T
G.head()



Products = list(G.columns)
Products.remove('Day of Order Date')

G.Datetime = pd.to_datetime(G['Day of Order Date'])

G.head(5)

def Prediction_Function(family_type, date_to_be_forecasted):
    # F : new data frame which will store a category which is to be predicted
    F = pd.DataFrame()

    F[family_type] = G[family_type]

    # F[family_type] = np.sqrt(F[family_type])

    F['date'] = G['Day of Order Date']

    F.sort_values(by='date')
    F = F.set_index('date')

    y = F[family_type]
    # y.plot(figsize=(15, 6))
    # plt.show()
    # print("PLOT FOR THE DATA")

    ########################################################################################################################

    import statsmodels.api as sm
    import pandas.util.testing as tm
    decomposition = sm.tsa.seasonal_decompose(y, model='additive')

    # fig = decomposition.plot()

    # plt.figure(figsize=(18,8))
    # plt.show()
    # print("Decomposition into various components like seasonality, skewdness")


    ########################################################################################################################

    # Arima (Auto Regressive Integrated Moving Average) : 

    # p : AR term
    # d : MA term
    # q : # of differencing required to make the data series stationary

    # if it has seasonal patterns then S-arima

    p = d = q = range(0, 2)
    import itertools
    pdq = list(itertools.product(p, d, q))
    s_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]


    # print('Examples of parameter combinations for Seasonal ARIMA...')
    # print('SARIMAX: {} x {}'.format(pdq[1], s_pdq[1]))
    # print('SARIMAX: {} x {}'.format(pdq[1], s_pdq[2]))
    # print('SARIMAX: {} x {}'.format(pdq[2], s_pdq[3]))
    # print('SARIMAX: {} x {}'.format(pdq[2], s_pdq[4]))

    import warnings
    warnings.filterwarnings("ignore")
    from statsmodels.tsa.arima_model import ARIMA

    # print(pdq)
    # print(s_pdq)

    ########################################################################################################################

    pdq_results = []

    for param in pdq:
        for param_seasonal in s_pdq:
            try:
                model = sm.tsa.statespace.SARIMAX(y, order=param, seasonal_order=param_seasonal, enforce_stationarity=False, enforce_invertibility=False) 
                results = model.fit() 
                # print(results.aic, param, param_seasonal)
                pdq_results.append([results.aic, param, param_seasonal])
            except:
                continue

    pdq_results.sort()
    # print(pdq_results)
    # print(pdq_results[0])

    p1 = pdq_results[0][1]
    p2 = pdq_results[0][2]

    # print(p1, p2)

    Data_to_fed = y['2021-08-09':]

    ########################################################################################################################

    Model = sm.tsa.statespace.SARIMAX(Data_to_fed, order=p1, seasonal_order=p2, enforce_stationarity=False, enforce_invertibility=False)
    # Model = sm.tsa.statespace.SARIMAX(Data_to_fed, order=p1, seasonal_order=p2)
    Results = model.fit()
    
    ########################################################################################################################

    # print(Results.summary().tables[0])
    # print(Results.summary().tables[1])
    # print(Results.summary().tables[2])


    ########################################################################################################################
    # I am seeing that the data is too peaked in the middle seeing by qq plot
    # Rectification : may use log 
    
    ########################################################################################################################

    # Model Diagostics
    # Results.plot_diagnostics(figsize=(16, 8))
    # Results.plot_diagnostics(figsize=(16, 8), lags = 15) # Number of lags to include in the correlogram. Default is 10.
    
    # plt.show()
    # print()

    #prediction 
    import datetime as dt

    #get predictions starting from the min date and calculate confidence intervals

    # Pred = Results.get_prediction(start=dt.datetime(2022, 5, 9), dynamic=False)
    Pred = Results.get_prediction(start='2021-08-09', dynamic=False)
    # Pred = Results.get_prediction(start='2022-08-09', dynamic=False)
    pred_ci = Pred.conf_int()

    # Methods
    # conf_int([alpha]) : Confidence interval construction for the predicted mean.
    # summary_frame([alpha]) : Summary frame of mean, variance and confidence interval.
    # t_test([value, alternative]) : z- or t-test for hypothesis that mean is equal to value

    # Properties
    # predicted_mean : The predicted mean 
    # row_labels : The row labels used in pandas-types.
    # se_mean : The standard deviation of the predicted mean
    # tvalues : The ratio of the predicted mean to its standard deviation
    # var_pred_mean : The variance of the predicted mean

    ########################################################################################################################

    # X_data = y['2021-08-09':].plot(label='Observed')

    # Pred.predicted_mean.plot(ax=X_data, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))

    # X_data.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.2)

    # X_data.set_xlabel('Date')
    # X_data.set_ylabel(family_type)

    # plt.show()
    # print()

    ########################################################################################################################

    y_forecasted = Pred.predicted_mean
    y_original = y['2021-08-09':]

    Mean_square_error = ((y_forecasted - y_original)**2).mean()

    # print(round(Mean_square_error, 4))
    MSE = Mean_square_error

    RMSE = MSE**0.5

    # print(family_type, "MSE :", MSE, "RMSE :", round(RMSE, 4))

    ########################################################################################################################
    # Further Prediction
    ########################################################################################################################

    # Pred_uc = Pred
    # pred_uc = results.get_forecast(steps=100) # steps 
    # pred_ci = pred_uc.conf_int() # for construction of the interval [ which shows future predicted values ]


    # # X_data = y.plot(label='Observed', figsize=(14, 7))

    # pred_uc.predicted_mean.plot(ax=X_data, label='Forecasting data values')

    # pred_dynamic = results.get_prediction(start='2022-01-01',dynamic=False, full_results=False)
    # pred_dynamic_ci = pred_dynamic.conf_int()


    # ax = y['2022':].plot(label='_Observed_')
    # pred_dynamic.predicted_mean.plot(ax=ax, label='Forecast', figsize=(20, 12), alpha=.7)

    # ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.2)

    # ax.set_xlabel('date')
    # ax.set_ylabel(family_type)

    # # plt.legend()
    # # plt.show()
    # # print()

    # # Future prediction and seasonality

    # pred_dynamic = results.get_forecast(steps=111)
    # pred_ci = pred_uc.conf_int()

    # # ax = y.plot(label='Observed', figsize=(20, 12))

    # pred_dynamic.predicted_mean.plot(ax=ax, label='Forecast')

    # ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.25)

    # ax.set_xlabel('date')
    # ax.set_ylabel(family_type)

    # # plt.legend()
    # # plt.show()
    # # print()

    Forecast = Results.predict(start=date_to_be_forecasted, end = date_to_be_forecasted)
    
    return Forecast[0]

    # print(Forecast[0])

def accuracy(family_type):
    # F : new data frame which will store a category which is to be predicted
    F = pd.DataFrame()

    F[family_type] = G[family_type]

    # F[family_type] = np.sqrt(F[family_type])

    F['date'] = G['Day of Order Date']

    F.sort_values(by='date')
    F = F.set_index('date')

    y = F[family_type]

    import statsmodels.api as sm
    import pandas.util.testing as tm
    decomposition = sm.tsa.seasonal_decompose(y, model='additive')

    p = d = q = range(0, 2)
    import itertools
    pdq = list(itertools.product(p, d, q))
    s_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

    import warnings
    warnings.filterwarnings("ignore")
    from statsmodels.tsa.arima_model import ARIMA

    pdq_results = []

    for param in pdq:
        for param_seasonal in s_pdq:
            try:
                model = sm.tsa.statespace.SARIMAX(y, order=param, seasonal_order=param_seasonal, enforce_stationarity=False, enforce_invertibility=False) 
                results = model.fit() 
                # print(results.aic, param, param_seasonal)
                pdq_results.append([results.aic, param, param_seasonal])
            except:
                continue

    pdq_results.sort()
    # print(pdq_results)
    # print(pdq_results[0])

    p1 = pdq_results[0][1]
    p2 = pdq_results[0][2]

    # print(p1, p2)

    Data_to_fed = y['2021-08-09':'2022-03-31']

    ########################################################################################################################

    Model = sm.tsa.statespace.SARIMAX(Data_to_fed, order=p1, seasonal_order=p2, enforce_stationarity=False, enforce_invertibility=False)
    # Model = sm.tsa.statespace.SARIMAX(Data_to_fed, order=p1, seasonal_order=p2)
    Results = model.fit()

    import datetime as dt

    Pred = Results.get_prediction(start='2022-04-01', end='2022-06-07', dynamic=False)
    pred_ci = Pred.conf_int()

    y_forecasted = Pred.predicted_mean
    y_original = y['2022-04-01':'2022-06-07']

    n = len(y_forecasted)

    maps = 0.0
    for i in range(0, n):
        maps += abs(y_forecasted[i] - y_original[i])/max(1, y_original[i])
    
    return maps/n;

A,p = 0,0
l = []
Family_members = ['AIR CARE', 'AIR CONDITIONER SERVICE', 'AUDIO', 'AUDIO ENHANCEMENT',
       'CNP IT', 'CNP MOBILITY', 'CONSUMABLES', 'COOKING APPLIANCE SERVICE',
       'COOKING APPLIANCES', 'DESKTOP', 'DISHWASHER SERVICE',
       'Delivery Charges', 'ENTERTAINMENT PROMOTION', 'ERECHARGE', 'FANS',
       'FOOD PRESERVATION', 'GAMING ACCESSORY', 'HARDWARE MOBILITY',
       'HIGH END TV', 'HOME APPLIANCES PROMOTION', 'HOME AUDIO SERVICE',
       'HOME CARE SERVICE', 'HOME VISUAL SERVICE', 'LAPTOP', 'LAPTOP SERVICE',
       'LAUNDRY & WASH CARE', 'LIFESTYLE IT', 'LIFESTYLE MOBILITY', 'LIGHTING',
       'MEDIAPLAYER', 'MUSICAL INSTRUMENTS (POWERED)',
       'Mobility (Jio Branded)', 'PERIPHERAL', 'PHOTOGRAPHY',
       'PHOTOGRAPHY WEARABLE', 'POWER', 'POWER ENTERTAINMENT', 'POWER HA',
       'PRINTER', 'Printer Services', 'REFRIGERATOR SERVICE',
       'SMALL DOMESTIC APPL', 'SMART HOME DEVICES', 'SOFTWARE', 'STORAGE',
       'TABLET', 'TABLET SERVICE', 'TECH ACCESSORIES IT',
       'TECH ACCESSORIES MOBILITY', 'TELECOM SERVICE', 'WASHER SERVICE',
       'WEARABLE DEVICE', 'WEARABLES SERVICE', 'WIRELESS PHONE',
       'WIRELESS PHONE SERVICE', 'BIG BAZAAR-PROD', 'DC / STORE CONSUMABLES',
       'DRYER SERVICE', 'GAMING SOFTWARE', 'HARDWARE HA']

print(len(Family_members))
for i in range(57, len(Family_members)):
    print(p, end=',')
    l.append(accuracy(Family_members[i]))
    p += 1

# print("Accuacy :", A/p)

print(l)
q = l
Family_members = ['AIR CARE', 'AIR CONDITIONER SERVICE', 'AUDIO', 'AUDIO ENHANCEMENT',
       'CNP IT', 'CNP MOBILITY', 'CONSUMABLES', 'COOKING APPLIANCE SERVICE',
       'COOKING APPLIANCES', 'DESKTOP', 'DISHWASHER SERVICE',
       'Delivery Charges', 'ENTERTAINMENT PROMOTION', 'ERECHARGE', 'FANS',
       'FOOD PRESERVATION', 'GAMING ACCESSORY', 'HARDWARE MOBILITY',
       'HIGH END TV', 'HOME APPLIANCES PROMOTION', 'HOME AUDIO SERVICE',
       'HOME CARE SERVICE', 'HOME VISUAL SERVICE', 'LAPTOP', 'LAPTOP SERVICE',
       'LAUNDRY & WASH CARE', 'LIFESTYLE IT', 'LIFESTYLE MOBILITY', 'LIGHTING',
       'MEDIAPLAYER', 'MUSICAL INSTRUMENTS (POWERED)',
       'Mobility (Jio Branded)', 'PERIPHERAL', 'PHOTOGRAPHY',
       'PHOTOGRAPHY WEARABLE', 'POWER', 'POWER ENTERTAINMENT', 'POWER HA',
       'PRINTER', 'Printer Services', 'REFRIGERATOR SERVICE',
       'SMALL DOMESTIC APPL', 'SMART HOME DEVICES', 'SOFTWARE', 'STORAGE',
       'TABLET', 'TABLET SERVICE', 'TECH ACCESSORIES IT',
       'TECH ACCESSORIES MOBILITY', 'TELECOM SERVICE', 'WASHER SERVICE',
       'WEARABLE DEVICE', 'WEARABLES SERVICE', 'WIRELESS PHONE',
       'WIRELESS PHONE SERVICE', 'BIG BAZAAR-PROD', 'DC / STORE CONSUMABLES',
       'DRYER SERVICE', 'GAMING SOFTWARE', 'HARDWARE HA']
print(len(Family_members))
for i in range(57, len(Family_members)):
    print(p, end=',')
    l.append(accuracy(Family_members[i]))
    p += 1

print(np.mean(l)*100)



date_to_be_forecasted = "2023-07-07"
family = 'FANS'
Value = Prediction_Function(family, date_to_be_forecasted)
print(Value)

import pickle
pickle.dump(Prediction_Function, open('model2.pkl','wb'))
model = pickle.load(open('model2.pkl','rb'))

#print(model.predict([[1.8]]))


#mp(family, date_to_be_forecasted)

Family_members = ['AIR CARE', 'AIR CONDITIONER SERVICE', 'AUDIO', 'AUDIO ENHANCEMENT',
       'CNP IT', 'CNP MOBILITY', 'CONSUMABLES', 'COOKING APPLIANCE SERVICE',
       'COOKING APPLIANCES', 'DESKTOP', 'DISHWASHER SERVICE',
       'Delivery Charges', 'ENTERTAINMENT PROMOTION', 'ERECHARGE', 'FANS',
       'FOOD PRESERVATION', 'GAMING ACCESSORY', 'HARDWARE MOBILITY',
       'HIGH END TV', 'HOME APPLIANCES PROMOTION', 'HOME AUDIO SERVICE',
       'HOME CARE SERVICE', 'HOME VISUAL SERVICE', 'LAPTOP', 'LAPTOP SERVICE',
       'LAUNDRY & WASH CARE', 'LIFESTYLE IT', 'LIFESTYLE MOBILITY', 'LIGHTING',
       'MEDIAPLAYER', 'MUSICAL INSTRUMENTS (POWERED)',
       'Mobility (Jio Branded)', 'PERIPHERAL', 'PHOTOGRAPHY',
       'PHOTOGRAPHY WEARABLE', 'POWER', 'POWER ENTERTAINMENT', 'POWER HA',
       'PRINTER', 'Printer Services', 'REFRIGERATOR SERVICE',
       'SMALL DOMESTIC APPL', 'SMART HOME DEVICES', 'SOFTWARE', 'STORAGE',
       'TABLET', 'TABLET SERVICE', 'TECH ACCESSORIES IT',
       'TECH ACCESSORIES MOBILITY', 'TELECOM SERVICE', 'WASHER SERVICE',
       'WEARABLE DEVICE', 'WEARABLES SERVICE', 'WIRELESS PHONE',
       'WIRELESS PHONE SERVICE', 'BIG BAZAAR-PROD', 'DC / STORE CONSUMABLES',
       'DRYER SERVICE', 'GAMING SOFTWARE', 'HARDWARE HA']

# for i in Family_members:
    # Prediction_Function(i)

# AIR CARE MSE : 21183335058643.676 RMSE : 4602535.7205
# AIR CONDITIONER SERVICE MSE : 123316042.37927368 RMSE : 11104.7757
# AUDIO MSE : 39309363367.189545 RMSE : 198265.8906
# AUDIO ENHANCEMENT MSE : 6922798808893.409 RMSE : 2631121.2076
# CNP IT MSE : 22242.509259843915 RMSE : 149.1392
# CNP MOBILITY MSE : 53523.626873642475 RMSE : 231.3517
# CONSUMABLES MSE : 15338117.379488252 RMSE : 3916.3909
# COOKING APPLIANCE SERVICE MSE : 15305.883282280573 RMSE : 123.7169
# COOKING APPLIANCES MSE : 938852125.1765873 RMSE : 30640.6939
# DESKTOP MSE : 2012995557.1807008 RMSE : 44866.419
# DISHWASHER SERVICE MSE : 0.09951151876546875 RMSE : 0.3155
# Delivery Charges MSE : 5381.527623541379 RMSE : 73.3589
# ENTERTAINMENT PROMOTION MSE : 17732790.560747992 RMSE : 4211.032
# ERECHARGE MSE : 855722532.5969683 RMSE : 29252.7355
# FANS MSE : 307955554.17458814 RMSE : 17548.6625
# FOOD PRESERVATION MSE : 2042351181661.7776 RMSE : 1429108.5269
# GAMING ACCESSORY MSE : 810739970.4669902 RMSE : 28473.4959
# HARDWARE MOBILITY MSE : 295785459.73483753 RMSE : 17198.4145
# HIGH END TV MSE : 5576520027641.804 RMSE : 2361465.6524
# HOME APPLIANCES PROMOTION MSE : 845504.1082343048 RMSE : 919.513
# HOME AUDIO SERVICE MSE : 50760.856888478025 RMSE : 225.3017
# HOME CARE SERVICE MSE : 396504.94510554767 RMSE : 629.6864
# HOME VISUAL SERVICE MSE : 18010381.005978886 RMSE : 4243.8639
# LAPTOP MSE : 853749668119.8507 RMSE : 923985.751
# LAPTOP SERVICE MSE : 50608.49443460146 RMSE : 224.9633
# LAUNDRY & WASH CARE MSE : 342156742334.0944 RMSE : 584941.6572
# LIFESTYLE IT MSE : 4127882.6369366213 RMSE : 2031.7191
# LIFESTYLE MOBILITY MSE : 2867297.6841404806 RMSE : 1693.3097
# LIGHTING MSE : 856546461.4395553 RMSE : 29266.815
# MEDIAPLAYER MSE : 53726217.356389426 RMSE : 7329.817
# MUSICAL INSTRUMENTS (POWERED) MSE : 2845481.1124221 RMSE : 1686.8554
# Mobility (Jio Branded) MSE : 2608162153611.6753 RMSE : 1614980.5428
# PERIPHERAL MSE : 2932335590.378207 RMSE : 54151.0442
# PHOTOGRAPHY MSE : 31060896586.71602 RMSE : 176241.0185
# PHOTOGRAPHY WEARABLE MSE : 210873273.08219883 RMSE : 14521.4763
# POWER MSE : 606557417975.1039 RMSE : 778817.962
# POWER ENTERTAINMENT MSE : 29199906.06467008 RMSE : 5403.6937
# POWER HA MSE : 9529185.424775867 RMSE : 3086.9379
# PRINTER MSE : 81565295451.81204 RMSE : 285596.3856
# Printer Services MSE : 108070.25282146975 RMSE : 328.7404
# REFRIGERATOR SERVICE MSE : 651941.2189528358 RMSE : 807.4288
# SMALL DOMESTIC APPL MSE : 1691458922722.652 RMSE : 1300561.0031
# SMART HOME DEVICES MSE : 215458890.013455 RMSE : 14678.518
# SOFTWARE MSE : 69950.13422763711 RMSE : 264.4809
# STORAGE MSE : 173803447571.49194 RMSE : 416897.4065
# TABLET MSE : 115724147681.09293 RMSE : 340182.5211
# TABLET SERVICE MSE : 8853.110482556436 RMSE : 94.091
# TECH ACCESSORIES IT MSE : 46230458.15059248 RMSE : 6799.2984
# TECH ACCESSORIES MOBILITY MSE : 1928514868.5007691 RMSE : 43914.8593
# TELECOM SERVICE MSE : 3.1591187010027424 RMSE : 1.7774
# WASHER SERVICE MSE : 1202221.500272414 RMSE : 1096.4586
# WEARABLE DEVICE MSE : 189109176089.14133 RMSE : 434866.8487
# WEARABLES SERVICE MSE : 78973.75552141499 RMSE : 281.0227
# WIRELESS PHONE MSE : 1716526687724572.8 RMSE : 41430987.0474
# WIRELESS PHONE SERVICE MSE : 26778489.241566308 RMSE : 5174.7936
# BIG BAZAAR-PROD MSE : 28459.144704506267 RMSE : 168.6984
# DC / STORE CONSUMABLES MSE : 474.96711033190235 RMSE : 21.7937
# DRYER SERVICE MSE : 0.004430310396342868 RMSE : 0.0666
# GAMING SOFTWARE MSE : 32682.428013787307 RMSE : 180.7828
# HARDWARE HA MSE : 637008.1193309585 RMSE : 798.1279

'''with open('/content/drive/MyDrive/Internship_2022/Files/DATA/real_full_data/DF_Model.pkl', 'wb') as f:
    pickle.dump(Prediction_Function, f)

with open('/content/drive/MyDrive/Internship_2022/Files/DATA/real_full_data/DF_Model.pkl', 'rb') as f:
    mp = pickle.load(f)
'''