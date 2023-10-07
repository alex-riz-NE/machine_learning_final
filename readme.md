**Predicting Stock Increase or Decrease Based on Publicly Available 10-k Business Reports**

**Alex Rizvanov**

Colby College

400 Mayflower Hill Dr,

Waterville, ME 04901

[afrizv22@colby.edu](mailto:afrizv22@colby.edu)

_Keywords_

10-k Report, Logistic Regression, K-Nearest Neighbors, Decision Tree

**Introduction**

Stock trading has turned into an algorithmic field with many students studying computer science and statistics to secure a job on Wallstreet. Firms are now looking to hire Quantitative Analysts (Quants) to find trends in stock data in order to maximize profits. Firms such as Quantitative Management Associates, Renaissance Technologies, Citadel, Jane Street, and Two Sigma handle between tens and hundreds of billions of dollars. These hedge funds hired many qualified statisticians that create extremely complex models in order to make profits and make sure the funds are not exposed to too much risk.

The average investor does not have access to the tools, resources, and knowledge that a large firm has. A complex algorithm incorporating years of data that backtests perfectly is not reasonable. If that investor does not want to invest in a hedge or index fund, they are stuck with picking individual stocks: commonly seen as a good way to lose money. Instead of doing individual analysis of thousands of stocks to find the diamonds in the rough a statistical model can do it for you. One way to do a simple model of stocks is to use the 10-k report from one year to the next in order to see if a trend can be found.

Every year every publicly traded business is required to produce a 10-k report to the SEC. This report is also available to the public so that investors have an understanding of the financial condition of the companies they might invest in. This document contains the history of the company, financial statements, earnings, money paid to executives, risk factors, product and services information, and many other important economic data. This report is a good indicator of how the company is doing at the moment.

Kaggle user Nicolas Carbone uploaded csv files of 10-K data from approximately 4000 companies every year from 2014 to 2018 using Financial Modeling Prep's API. The 2018 data is used in order to make the model and the 2017 data is used in order to see if the models work on data from another year. For each of the years, the API scrapes more than 200 variables from the 10-K statement. Some of these variables are dependent on each other which would hurt the model due to the linear dependencies that would be in the data. I took those variables out. Here is a list of variables that were taking into consideration the model:

"Company Name", "Revenue", "Revenue Growth", "Gross Profit" , "R&D Expenses", "Operating Expenses", "Preferred Dividends", "Net Income Com", "EPS", "Weighted Average Shs Out", "Dividend per Share", "Profit Margin", "Free Cash Flow margin", "Total assets", "Total debt", "Deferred revenue", "Total liabilities", "Other comprehensive income", "Other comprehensive income", "Net Debt", "Other Assets", "Other Liabilities", "Depreciation & Amortization", "Operating Cash Flow", "Capital Expenditure", "Acquisitions and disposals", "Investing Cash flow", "Dividend payments", "Free Cash Flow", "Net Cash/Marketcap", "priceBookValueRatio", "priceEarningsRatio", "Dividend Yield", "grossProfitMargin", "returnOnEquity", "debtRatio", "debtEquityRatio", "Revenue per Share", "Market Cap", "PE ratio", "Earnings Yield", "R&D to Revenue", "Graham Number", "ROE", "Gross Profit Growth", "5Y Revenue Growth (per Share)", "3Y Revenue Growth (per Share)", "5Y Net Income Growth (per Share)", "3Y Net Income Growth (per Share)","5Y Dividend per Share Growth (per Share)", "3Y Dividend per Share Growth (per Share)", "Inventory Growth", "Asset Growth", "Book Value per Share Growth", "Debt Growth", "R&D Expense Growth", "SG&A Expenses Growth", "Sector", "201X PRICE VAR [%]", and "Class".

While there are only around a fourth of the variables remain from the original list, this list still covers most of the basics of a 10-k, such as, Revenue, Profit, R&D, Debt, Dividends, Cash Flow, PE, ROE, and Growth. The only variable not found in the 10-k is the Class variable. Nicolas Carbone determined the Class variable with the "201X PRICE VAR" variable. If it increased for the company the class of that company is one; it would be zero if it decreased.

**Exploratory Data Analysis:**

In order to see how the different predictors relate to the Class response variable a correlation test was done. In general, the predictors did not have a large correlation with Class. Some of the predictors of note though would be Net Income, Dividend per Share, Net Income Growth per Share (x5 Years), Net Income per Share (x5 Years), Dividend Growth per Share (x5 Years), and Gross Profits with correlations of 0.097, 0.131, 0.072, 0.107, 0.107, and 0.079 respectively (Appendix Fig1). It may be interesting to look at these predictors further, but they do not seem to explain much of the variance in the Class response variable. It seems like all of the variables should be included in models due to the low overall correlation with the response variable.

This data set incorporates many different sized companies from different industries with different structures. They all are set up in different way and have different operation costs. In order to see how the data looks, histograms were made for all predictors. Across all of the predictors it seems like the distributions are centered at zero. This is partially due to imputing all null values with 0 because a missing value means that the company's structure doesn't support that certain predictor, such as dividends.

For most of the predictors it makes sense to center around zero because on average companies don't make or lose that much, but there is an even distribution of increase and decrease between the business. This can be seen in the per share growth of Income, Revenue, and Dividend averaged over the last five years. When dividing by the amount of shares there are, most companies don't have a large increase or decrease in revenue or income and some break even (Fig 1 a,b). When looking at dividends it makes sense that some companies would have 0 dividend growth because some companies do not have dividends (Fig1 c).

The companies' average revenue growth per share over the last five years looks to have a fairly normal distribution (Fig 1 a) , but net income and dividend growth seems to have most of the data close to 0 (Fig c). One of these two patterns is seen in all of the predictors. The Free Cash Flow, Net Debt, Gross Profit, Research and Development Expense Growth, and Book Value per Share growth predictors all have a fairly normal distribution (Fig2 a, b, c, d, e). On the other hand, there are plenty of predictors like capital expenditure and Price Book Value Ratio (Fig3 a,b) which have a very concentrated distribution. These values probably wont be useful in any model since they are very similar for all companies.

**1a.**

![](RackMultipart20231007-1-y5xdh6_html_df66b1349faa42d5.png)

**1b.**

![](RackMultipart20231007-1-y5xdh6_html_66728e46221bb0c8.png)

**1c.**

![](RackMultipart20231007-1-y5xdh6_html_80a2d9841cf21043.png)

**Figure 1 Histogram of predictors from 10-k report data that are averaged over the last 5 years.** (a) Revenue growth per share, (b) Net income growth per share, (c) Dividend Growth Per Share averages from the last 5 years taken from the 2018 10-k report data using Financial Modeling Prep's API (n=4374).

**2a. 2b. 2c.**

![](RackMultipart20231007-1-y5xdh6_html_66fc010f1028bed1.png) ![](RackMultipart20231007-1-y5xdh6_html_18e12b8f4a21d872.png) ![](RackMultipart20231007-1-y5xdh6_html_467c9b5d614d3ea9.png)

**2d. 2e.**

![](RackMultipart20231007-1-y5xdh6_html_c2c64f3de5cfd667.png) ![](RackMultipart20231007-1-y5xdh6_html_eac09b97f8d77e94.png)

**Figure 2 Histogram of predictors from 10-k report data that have a fairly normal distribution.** (a) Free Cash Flow, (b) Net Debt, (c) Gross Profit, (d) Research and Development Expense Growth, (e) Book Value per Share Growth taken from the 2018 10-k report data using Financial Modeling Prep's API (n=4374)

**3a. 3b.**

![](RackMultipart20231007-1-y5xdh6_html_ff9819e0a1f1c475.png) ![](RackMultipart20231007-1-y5xdh6_html_cddbcedaad38419d.png)

**Figure 3 Histogram of predictors from 10-k report data that do not have a normal distribution.** (a) Capital Expenditure, (b) Price Book Value Ratio taken from the 2018 10-k report data using Financial Modeling Prep's API (n=4374).

**Classification**

**Logistic Regression**

After separating data into training and testing data subset selection was done using forward selection in order to find the optimum number of predictors to use for logistic regression. The model with the smallest Cp value has 24 predictors (Fig 4a). The model with the smallest BIC value is 9 (Fig 4b). The model with the highest adjusted value is 47 (Fig 4c). While looking at these graphs a value of 18 was determined to optimize the combined errors of the three different error techniques.

After finding the optimal model, a logistic regression model was made with the best 18 predictors. The model was trained on the 201810-k training data and tested on the 2018 10-k testing data. It had a misclassification error of 29.21%. This is much less than 50% for randomly guessing. The logistic regression model trained on the 2018 10-k dataset was then tested on the whole 2017 10-k dataset in order to see if the model could predict if stock prices rise or fell. This logistic regression when predicting the 2017 dataset had an error rate of 69.13%. The model did a lot better predicting when the stock went up (1) and did a lot worse when the stock went down (0) (Table 1).

**4a. 4b. 4c.**

![](RackMultipart20231007-1-y5xdh6_html_344aeda2f5db6a10.png)

**Fig 4 Cp, BIC, and adjusted for the 2018 10-k report company data set are shown for models containing the best n number of Predicts chosen by forward stepwise selection.** (a) Cp of models with an increasing number of predictors with the minimum value shown in red, (b) BIC of models with an increasing number of predictors with the minimum value shown in red, (c) adjusted of models with an increasing number of predictors with the maximum value shown in red.

![](RackMultipart20231007-1-y5xdh6_html_38f1349b75c235f4.png)

**Table 1. Misclassification table of the logistic regression model predicting the 2017 10-k dataset, trained on the 2018 training data 10-k dataset. The 0 category represents the company's stock decreasing price the next year, while10 category represents the company's stock increasing price the next year**

**K-Nearest Neighbors (KNN)**

First, dummy variables for the non-numeric predictions were created. Then the datasets were normalized in order to account for differences in range of predictors. Next, the optimal k was found by picking the lowest misclassification rate of models with k values of 3, 4, 5, 6, 7, 8, 9, 10, 25, 50, and 100 (FNN package v1.1.3). These models included all of the predictors and were trained using the train 2018 10-k data and they were tested on the test 2018 10-k data. The model optimal model, with the lowest misclassification, had a k of 25 (Fig 5).

After the optimal k the optimal model was used to predict the entire 2017 10-k data set (FNN package v1.1.3). It had a misclassification rate of 62.83%, which is more than the 50% for randomly guessing. This is slightly better than the error found from the logistic regression model, but not much better. Similar to the logistic regression, the KNN model predicting the 2017 10-k dataset was better at predicting when the stock went up (1) and did a lot worse when the stock went down (0) (Table 2).

![](RackMultipart20231007-1-y5xdh6_html_5b323617fd0a26ae.png)

**Fig 5. Misclassification rate for KNN models increasing K values.** The model was trained on the 2018 train 10-k dataset and is predicting the 2018 test 10-k dataset. The red dot shows the minimum misclassification rate.

![](RackMultipart20231007-1-y5xdh6_html_517118188ff5e9fb.png)

**Table 2. Misclassification table for the KNN model predicting the 2017 10-k dataset, trained on the 2018 training data 10-k dataset.**

**Decision Trees**

The Class response variable was made into a factor which was two levels: "Up" and "Down". Then a decision tree with all predictors as an input was made. Only the "Earnings.Yield", "Market.Cap", "Total.liabilities", "Total.assets" and "Dividend.GPS.3Year" predictors were used. In this model, the "Earnings.Yield" predictor was the most important predictor in splitting up the data (Fig6). When used to predict the 2018 test 10-k data, the model had a misclassification rate of around 28.12%. This model was then used to predict the entire 2017 10-k dataset, and the misclassification rate was 53.04%. This is a much better misclassification rate than the two previous models, but still greater than the randomly selected 50%. This model was still better at predicting the company if it increased price the next year, but it was a lot better at predicting when the market went down than the two previous models at 39.17% compared to 18.88% for KNN and 5.31% for logistic regression.

Boosting was then done in order to see if the model could be improved even further. The shrinkage parameter was found by creating models containing a sequence of values between 0.0001 and 5 and then choosing the model that had the smallest misclassification rate. The model with the smallest misclassification rate had a lambda of 0.4201 and a misclassification rate of 57.48%. Boosting did not seem to do better than the original tree, in fact it did worse. This may be because of the characteristics of the dataset. Most of the predictors are not strongly correlated with the response variable "Class." This means that the model does not get the usual benefit of slowly creating the tree iteration after iteration.

![](RackMultipart20231007-1-y5xdh6_html_b783c6df614e8131.png)

**Fig 6. A tree corresponding predicting the Class response variable in the 2017 10-k dataset.** The model was trained on the 2018 train 10-k dataset and was not pruned.

![](RackMultipart20231007-1-y5xdh6_html_de622fca50a8e8a9.png)

**Table 3. Misclassification table for the Decision Tree model predicting the 2017 10-k dataset, trained on the 2018 training data 10-k dataset.**

**Best model**

The logistic regression model had a misclassification rate of 69.13%. The KNN model had a misclassification rate of 62.83%. The Decision Tree model did the best, but still had a misclassification rate of 53.04%. All of these models all had misclassification rates less than 50%, which is worse than randomly guessing. Digging in to the results a little further shows that that the models do better predicting when the stock goes down as opposed to when it goes up. This is because the models all predict that there are going to be more stocks increasing in price than those decreasing. However, when looking at the values in the first row of all three misclassification tables, there are more instances of the stock going down.

This may be due to a difference in datasets. The 2018 dataset appears to document a bull market in which most stock prices rise, while the 2017 dataset appears to document a bear market where most stock prices fall. Since the models are trained from a subset of 2018, in would be hard to properly fit 2017 because the model was trained to predict more increases.

The model that did the best, the tree model, did a good job incorporating both increases and decreases. While the model did not do as good of a job as the other two in predicting when a stock price would rise, it did do a better job in predicting when the stock would go down. It was more flexible and allowed for nonlinearity.

This model not only was the most accurate, but it was also the most easily analyzed. It is clear that the -predictors "Earnings.Yield", "Market.Cap", "Total.liabilities", "Total.assets" and "Dividend.GPS.3Year" all play a role in determining the splits in the tree.

**Conclusion**

Making algorithms to predict stock prices is not an easy task. Even with all the statisticians and technologies that large quantitative hedge funds have, their models still sometimes do not do a good job predicting the market. That is because the stock market does not follow any particular pattern. There are just too many firms, people, changes in businesses, and current events to accurately predict everything.

None of the models performed well. They all had misclassification rates higher than 50%, which is not better than randomly choosing. This does not seem to be a problem with the procedure, however. With the data from the 10-k, classification is almost impossible. Training the data with the 2018 data did end up producing low misclassification rates when testing on the test 2018 data, but the data used to create the model means that the model would have high variance. It is too dependent on the data being put into it, which is not a good thing in this case. None of the predictors were remotely correlated to "Class". A model can't just magically create a trend, there needs to already be on in the data. Furthermore, most of the predictors had a majority of their values around 0. While none of the models required the data to have a normal distribution, having a lot of values be small makes it hard to find trends. Lastly, the market is just so different from year to year. The data sets were basically inverse. All the stocks went up in 2018 while they all went down in 2017. This classification might have been able to work if trained with a mix of data from a lot of different years.

Having a model that has a 53.04% misclassification is not terrible. This model is not meant to be an automatic trading tool. It was. Made in order to see what companies might go up in the next year. In that regard it did very well, and could be a good tool to start with when trying to find good stocks to invest in.

Appendix

A

![](RackMultipart20231007-1-y5xdh6_html_e5296b09d3500c98.gif)

**B (Code)**

### Load In data

### df2018 \<- **read.csv** ("trimmed\_2018data.csv")
names2018 = **names** (df2018)
_#names2018_
df2018 =df2018[**-** 60]
df2017 \<- **read.csv** ("trimmed\_2017data.csv")
names2017 = **names** (df2017)
_#names2017_
df2017 =df2017[**-** 60]

_#Make sure all the companies are the same for each year_
companies2017\<- **unlist** (df2017["Company.Name"])
companies2018 \<- **unlist** (df2018["Company.Name"])
companies.in.common = **intersect** (companies2017,companies2018)
idx2017 =companies2017 **%in%** companies.in.common
df2017=df2017[idx2017 **==** TRUE,][**-** 1]
idx2018 =companies2018 **%in%** companies.in.common
df2018=df2018[idx2018 **==** TRUE,][**-** 1]

### ###Finding correlations between predictor and response variables

### cor = **rep** (NA, **length** ( **ncol** (df2018) **-** 2))
nameVal = **rep** (NA, **length** ( **ncol** (df2018) **-** 2))
**for** (col **in** 1 **:** ( **ncol** (df2018) **-** 2)) {
 cor[col] = **cor** (df2018[,col],df2018[,59] )
 nameVal[col] = **names** (df2018)[col]
}

correlationsOfPredictorsToClass= **data.frame** ( names = nameVal, cor = cor )
**library** (gridExtra)
**pdf** ("correlationsOfPredictorsToClass.pdf", height=20, width=8.5)
**grid.table** (correlationsOfPredictorsToClass)
**dev.off** ()

**hist** (df2018[,"X5Y.Revenue.Growth..per.Share."],main ="Revenue Growth Per Share (5 years)", xlab ="Per Share ($)", breaks =100, xlim = **c** ( **-** 2,7))

**hist** (df2018[,"X5Y.Net.Income.Growth..per.Share."],main ="Net Income Growth Per Share (5 years) ", xlab ="Per Share ($)", breaks =100)

**hist** (df2018[,"X5Y.Dividend.per.Share.Growth..per.Share."],main ="Dividend Growth Per Share (5 years)", xlab ="Per Share ($)", breaks =30, xlim = **c** ( **-** 2,6))

**hist** (df2018[,"Free.Cash.Flow"] **/** 1000000000,main ="Free Cash Flow", xlab ="($ Billion)", breaks =75, xlim = **c** ( **-** 10,20))

**hist** (df2018[,"R.D.Expense.Growth"],main ="Research and Development Expense Growth", xlab ="($)", breaks =100, xlim = **c** ( **-** 5,15))

**hist** (df2018[,"Net.Debt"] **/** 1000000000,main ="Net Debt", xlab ="($ Billion)", breaks =50, xlim = **c** ( **-** 100,300))

**hist** (df2018[,"Gross.Profit"] **/** 1000000000,main ="Gross Profit", xlab =" ($ Billion)", breaks =50, xlim= **c** ( **-** 10,40))

**hist** (df2018[,"Capital.Expenditure"] **/** 1000000,main ="Capital Expenditure", xlab ="($ Million)", breaks =75, xlim = **c** ( **-** 8000, 1000))

**hist** (df2018[,"Book.Value.per.Share.Growth"],main ="Book Value per Share Growth", breaks =200,xlim = **c** ( **-** 10,50), xlab ="($)" )

**hist** (df2018[,"priceBookValueRatio"] **/** 1000000,main ="Price Book Value Ratio", breaks =500, xlab =" ($ Million)" , xlim= **c** ( **-** 2,5))

### Normalize Data

### _#normalize the data_
normalize\<- **function** (x){
**return** ((x **-**** min**(x)**/**(**max**(x)**- ****min** (x))))
}

df\_normalized2018 = **as.data.frame** ( **lapply** (df2018[**-**** c**(58,59)],normalize))
df\_normalized2018 **$** Class=df2018 **$** Class
df\_normalized2017 = **as.data.frame** ( **lapply** (df2017[**-**** c**(58,59)],normalize))
df\_normalized2017 **$** Class=df2017 **$** Class

### Making training and testing data

**set.seed** (1)
trainID= **sample** (1 **:**** nrow**(df\_normalized2018),**nrow**(df\_normalized2018)**/**2)
trainData2018 =df2018[trainID,]
testData2018 =df2018[**-** trainID,]
trainData2018\_normalized =df\_normalized2018[trainID,]
testData2018\_normalized =df\_normalized2018[**-** trainID,]
trainID2017= **sample** (1 **:**** nrow**(df\_normalized2017),**nrow**(df\_normalized2017)**/**2)
trainData2017\_normalized =df\_normalized2017[trainID2017,]
testData2017\_normalized =df\_normalized2017[**-** trainID2017,]
trainData2017 =df2017[trainID2017,]
testData2017 =df2017[**-** trainID2017,]

### Subset selection

**set.seed** (1)
reg.forward = **regsubsets** (Class **~**. , data=trainData2018, method="forward", nvmax=50)

reg.forward.summary = **summary** (reg.forward)

**par** (mfrow = **c** (1, 3))

min.cp \<- **which.min** (reg.forward.summary **$** cp)

min.bic \<- **which.min** (reg.forward.summary **$** bic)

max.adjr \<- **which.max** (reg.forward.summary **$** adjr2)

**plot** (reg.forward.summary **$** cp,xlab="Number of Predictors", ylab="Cp")
**points** (min.cp, reg.forward.summary **$** cp[min.cp], col="red", cex =2, pch =20)
**plot** (reg.forward.summary **$** bic,xlab="Number of Predictors", ylab="Bic")
**points** (min.bic, reg.forward.summary **$** bic[min.bic], col="red", cex =2, pch =20)
**plot** (reg.forward.summary **$** adjr2,xlab="Number of Predictors", ylab="Adjr")
**points** (max.adjr, reg.forward.summary **$** adjr2[max.adjr], col="red", cex =2, pch =20)

numOfPreds =18
predictors =reg.forward **$** xnames[2 **:** numOfPreds **+** 1] _#Chose 20 predictors_
fmla \<- **as.formula** ( **paste** ("Class ~", **paste** (predictors, collapse ="+")))

### Logistic Regression on just the 2018 data

**library** (ISLR)

my.mlr = **glm** (fmla, data=trainData2018, family ="binomial")

prob.train = **predict** (my.mlr ,testData2018 , type="response")
predict01.mlr = **ifelse** (prob.train **\>** 0.5, 1, 0)
Actual \<-testData2018 **$** Class
Predicted \<-predict01.mlr
predict01.mlr = **ifelse** (prob.train **\>** 0.5, 1, 0)

1 **-**** mean**(Actual**==**Predicted)

### Testing if the model would work on the 2017 data set

my.mlr = **glm** (fmla, data=trainData2018, family ="binomial")

prob.train = **predict** (my.mlr ,df2017 , type="response")
predict01.mlr = **ifelse** (prob.train **\>** 0.5, 1, 0)
Actual \<-df2017 **$** Class
Predicted \<-predict01.mlr
**table** (Actual, Predicted)

1 **-**** mean**(df2017**$ **Class** ==**predict01.mlr)

### Knn

**set.seed** (1)

**library** (class)

**library** (dplyr)

**library** (FNN)

**library** (psych)

k = **c** (3, 4,5,6,7,8,9, 10,25,50,100)
trainData2018[58] = **as.data.frame** ( **dummy.code** (trainData2018 **$** Sector))

testData2018[58] = **as.data.frame** ( **dummy.code** (testData2018 **$** Sector))

df2017[58] = **as.data.frame** ( **dummy.code** (df2017 **$** Sector))

make\_knn\_pred = **function** (k) {
 pred =FNN **::**** knn.reg**(train = trainData2018,
test = testData2018,
y = trainData2018 **$** Class, k = k) **$** pred
 pred = **ifelse** (pred **\>** 0.5, 1, 0)
 act =testData2018 **$** Class
**table** (pred, act)
**print** (1 **-**** mean**(pred**==**act))
}


knn\_trn\_errorRate = **sapply** (k, make\_knn\_pred)

best\_k =k[**which.min** (knn\_trn\_errorRate)]

**min** (knn\_trn\_errorRate)

**plot** (k,knn\_trn\_errorRate , xlab ="K" , ylab ="Misclassification Rate")
**points** (best\_k, **min** (knn\_trn\_errorRate), col="red", cex =2, pch =20)

prob\_knn\_2017 = **knn.reg** (train = trainData2018, test = df2017, y = trainData2018 **$** Class, k = best\_k) **$** pred
predicted = **ifelse** (prob\_knn\_2017 **\>** 0.5, 1, 0)
actual =df2017 **$** Class
**table** (actual, predicted)

**mean** (actual **!=** predicted )

### Tree

**library** (tree)

trainData2018 **$** Class = **as.factor** ( **ifelse** (trainData2018 **$** Class **==** 1, "Up", "Down"))
testData2018 **$** Class = **as.factor** ( **ifelse** (testData2018 **$** Class **==** 1, "Up", "Down"))
df2017 **$** Class = **as.factor** ( **ifelse** (df2017 **$** Class **==** 1, "Up", "Down"))
df2018 **$** Class = **as.factor** ( **ifelse** (df2018 **$** Class **==** 1, "Up", "Down"))

**names** (trainData2018)[**which** ( **names** (trainData2018) **==**"X3Y.Dividend.per.Share.Growth..per.Share.")] \<- "Dividend.GPS.3Year"
trainData2018 = **data.frame** (trainData2018)
**names** (testData2018)[**which** ( **names** (testData2018) **==**"X3Y.Dividend.per.Share.Growth..per.Share.")] \<- "Dividend.GPS.3Year"
testData2018 = **data.frame** (testData2018)
**names** (df2017)[**which** ( **names** (df2017) **==**"X3Y.Dividend.per.Share.Growth..per.Share.")] \<- "Dividend.GPS.3Year"
trainData2017 = **data.frame** (trainData2017)

tree.2018 \<- **tree** (Class **~**., data = trainData2018)
**summary** (tree.2018)

**pdf** ("treeBasic.pdf")
**plot** (tree.2018,margin)
**text** (tree.2018, pretty =1, cex =1.2)
**dev.off** ()

predicted =tree.pred = **predict** (tree.2018, newdata = testData2018, type="class")
actual =testData2018 **$** Class
**table** (predicted, actual)

**mean** (predicted **!=** actual)

predicted =tree.pred = **predict** (tree.2018, newdata = df2017, type="class")
actual =df2017 **$** Class

**mean** (predicted **!=** actual)

### Tree Boosting

**library** (gbm)

lambdas = **seq** (0.0001,0.5,0.01)
missclassificationRate = **rep** (NA, **length** (lambdas))
**for** ( i **in** lambdas){
 boost.hit = **gbm** (Class **~**., data=trainData2018, distribution="bernoulli", n.tree=250,
shrinkage= i, interaction.depth =5)
 prob.train = **predict.gbm** (boost.hit, newdata = df2017, n.trees=250, type ="response")
 pred = **ifelse** (prob.train **\>** 0.5, 1, 0)
 missclassificationRate[**which** (i **==** lambdas)]=1 **-**** mean**(pred**== **df2017** $**Class)
}

**plot** (lambdas, missclassificationRate)
**points** (max.adjr, reg.forward.summary **$** adjr2[max.adjr], col="red", cex =2, pch =20)

best\_lambda =lambdas[**which.min** (missclassificationRate)]
lowestError = **min** (missclassificationRate)
