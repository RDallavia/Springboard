# Ticket Pricing at Big Mountain Resort
![](images1/overlook_postcard.jpg)
***The Overlook Hotel from Stephen King's "The Shining"***
***(Source: [Etsy](https://www.etsy.com/listing/729040276/the-overlook-hotel-postcard-stephen-king))*** 

Running a business can be boiled down to three simple questions: 

1. What should we invest in?
2. How should we pay for it?
3. How should we return value to stakeholders?

Management at Big Mountain Resort (a/k/a "the Company", "the Client", "the resort") was struggling with questions one and three. The Resort had recently added a chairlift, which increased visitor mobility around the complex. The lift also increased operating costs by $1.54 million in the current season. 

There's nothing wrong with an investment in a lift at a ski resort per se, but management was uncertain which asset types actually added value to the customer's experience of Big Mountain. Restated, the Company made an investment they were unsure visitors would value. Moreover, management had yet to determine whether or not the resort's asset portfolio (e.g., lifts, runs, amenities, etc.) could support the current ticket price or, potentially, a higher one.

We were called in to determine which assets were valued most by customers and resolve the question of whether or not the resort's tickets were appropriately priced.
# The Data
Data were provided to us in a flat file (CSV format). The file contained 330 rows and 27 columns. Each row represented a U.S. ski resort. The features (i.e., columns) below were provided to support our analysis.
![](images1/Features.png) State-level population data were pulled from Wikipedia.com.   

# [Data Wrangling/Cleaning](https://github.com/RDallavia/Springboard/blob/main/ProductPricingModel/Notebooks/02_data_wrangling.ipynb)
Data for Big Mountain was complete in all respects. The biggest surprise found during the wrangling process was the existence of two potential target variables: `AdultWeekday` and `AdultWeekend`. The `AdultWeekday` variable was dropped, as it was missing slightly more data points than `AdultWeekend`. The loss of information was minimal. Only 47 were missing both types of ticket data, and many resorts offered ticket prices that were the same during both the week and weekend.

Beyond the existence of two target variables, the key takeaways from the data wrangling phase of our study were fairly commonplace. `fastEight` was dropped as a potential feature because over 50% of our resorts did not have a value for this particular variable. The presence of `Region` and `state` variables caused some initial confusion, however, it soon became clear that the two were not isomorphic. A resort's `Region` and `state` differed 33 times. California had the most regions, but New York was the industry leader in total number of resorts measured by region or state. Montana ranked 11th by `Region` and 12th by `state`.

A boxplot revealed average ticket price variance was significant when viewed across state lines. The distribution of ticket prices for the `AdultWeekend` tickets in Montana was fairly narrow and mirrored the distribution for weekday tickets. This stands in contrast to ski markets including, but not limited to, Utah, Vermont, and California.

Most features exhibited a distinctive positive skew owing to the inability of the variable to take on a value less than zero. Erroneous data appeared as outliers in the `SkiableTerrain_ac` and `yearsOpen` columns, as well as `SnowMaking_ac`. `SkiableTerrain_ac` was corrected using data from the internet. The remaining outliers mentioned were dropped.

The following features were engineered by summation within each state:
* TerrainParks
* SkiableTerrain_ac
* daysOpenLastYear
* NightSkiing_ac

Only after reviewing all data did we drop rows containing no price data. This is a conservative approach to data wrangling. Rows missing price data may have data pertaining to other features than could provide insight when coompleting the wrangling process.

State population data was pulled from [Wikipedia](https://simple.wikipedia.org/w/index.php?title=List_of_U.S._states&oldid=7168473).
When aggregated, it was apparent missing data fell into quantiles, suggesting human intervention. This was no surprise, given that our client is fictional.
# [Data Exploration](https://github.com/RDallavia/Springboard/blob/main/ProductPricingModel/Notebooks/03_exploratory_data_analysis.ipynb)
We entered the data exploration phase of study with a pared-down data set (a/k/a "design matrix") of 278 rows and 25 features including `AdultWeekend` ticket prices, which will be extracted from our matrix and isolated before modelling. Additionally, we had a design matrix of state-level data containing summations for the following variables:
* TerrainParks
* SkiableTerrain_ac
* daysOpenLastYear
* NightSkiing_ac

The summation process gave rise to the following features: 
* state 
* resorts_per_state 
* state_total_skiable_area_ac
* state_total_days_open
* state_total_terrain_parks 
* state_total_nightskiing_ac

Combining our data sets provided us with valuable insights about our resort, Montana, and their standings relative to their competitors.

Montana claimed the top spot in skiable acreage in its immediate vicinity defined here as including Idaho, Wyoming, and South Dakota. If we broaden our perspective, Colarado and Utah come in first and second, respectively, while Montana ranks third among the top 20 states with the most skiable area.

Skiable area didn't predict the resort count by state. The three states with the most resorts were New York, Michigan, and Colarado. Montana ranked 12th in our 35 state lineup. New York was also first in night skiing whereas Montana came in 8th.

The total days open seems to bear some resemblance to the number of resorts. This is reasonable. The more resorts in a state the greater the chances a resort will be open at any point in the season, thus, the state will have more ski days.

New Hampshire, however, made it into the top 5, despite being a small state that didn't make it into the top 5 for resorts per state. This is likely due both to its average annual snowfall of 60" to 150" and its proximity to New York and, oft overlooked, Connecticut. Montana took our 15th spot.

We looked at resorts per 100k people and 100 square miles. Due to outliers and the data's inability to go below zero, both distributions had a distinct positive skew. 

Many states had between 8 and 11 resorts per 100k people, though a non-trivial number had between 1 and 4. Vermont, Wyoming, New Hampshire, and Montana ranked first through fourth, respectively, regarding resorts per 100k.  

When looking at resorts per 100k square miles, most states have between 3 and 9 resorts. New Hampshire dominated the ranking, while Montana failed to make our top 10.

We performed [Principal Component Analysis (PCA)](https://en.wikipedia.org/wiki/Principal_component_analysis) on our data. The first two composite features accounted for over 75% of the data's variance, and the first four components accounted for over 95%. Plotting the first two principal components against one another revealed no particular pattern with respect to price or any other variable. Our PCA did suggest `resorts_per_100kcapita` and `resorts_per_100ksq_mile` had an outsized influence on the data and may explain why states like Vermont and New Hampshire were outliers.  

A simple bar chart of average weekend prices indicated resorts either tended to cluster between $40 to $60 and around $80.

We engineered resort level features to augment our design matrix. The features were as follows:
* ratio of resort skiable area to total state skiable area
* ratio of resort days open to total state days open
* ratio of resort terrain park count to total state terrain park count
* ratio of resort night skiing area to total state night skiing area

A heatmap revealed AdultWeekend ticket price, we see quite a few reasonable correlations. `fastQuads` stood out, along with `Runs` and `Snow Making_ac`. Visitors seem to value more "guaranteed" snow, which would drive up resort costs and ticket prices. Of the new features,`resort_night_skiing_state_ratio` seemed the most correlated with ticket price. If this is true, then perhaps seizing a greater share of night skiing capacity means the resort can charge more, as it should.  Runs and total_chairs were also well correlated with ticket price, and `vertical_drop seems` to be a selling point that raised ticket prices as well.
# [Model Preprocessing and Training](https://github.com/RDallavia/Springboard/blob/main/ProductPricingModel/Notebooks/04_preprocessing_and_training.ipynb)
We entered the 4th stage of our process with a design matrix of 277 rows and 25 columns, after having removed rows pertaining to our client's resort. We performed a standard 70%/30% train/test split on the data set and withheld the `AdultWeekend` column as the target. 

We separated out the columns `Name`, `state`, and `Region`, which left us with only numeric data in our design and testing matricies. 

The grand mean was approximately 63.84. This is the mean of the target variable and our baseline predictor. If our model cannot out perform this, then there is little use in relying on our model.

A commonplace metric for the extent to which a model outperforms the baseline prediction is $R^{2}$. As we observed, the $R^{2}$ for a model no better than the baseline is 0, while a perfect model will yield an $R^{2}$ of 1. Mean squared error and mean absolute error were also explored.

We experiemented with filling missing values with column medians and used the standard scaler to put all data on equal footing. We fit a linear model to the data and discovered our test set results were better than our baseline (~73 versus ~63). The mean absolute error indicates we'd be within ~9.00 of the actual ticket price using our model.

We repeated the above steps using means to fill in missing feature values. The results were largely the same as when we used the median.

We defined a pipeline to capture our process in the following way:
```Python
pipe = make_pipeline(
    SimpleImputer(strategy='median'), 
    StandardScaler(), 
    LinearRegression()
)
```
After fitting and predicting using our pipeline like so, 
```Python
pipe.fit(X_train, y_train)
y_tr_pred = pipe.predict(X_train)
y_te_pred = pipe.predict(X_test)
```
it became clear that our pipeline was working as expected, producing mean absolute error we had seen before. Trying to improve the model by selecting the k best predictors where k=10 and k=15 did not improve our predictive ability. We ultimately switched to a random forest model and turned to 5-fold, Grid search cross validation. The results indicated our random forest model would be within approximately $1 of the actual price -- a significant improvement over our linear model. Thus, we chose it as the model with which we'll go forward. Interestingly, our random forest model agreed with our linear model insofar as it identified the following as the most important features:
* fastQuads
* Runs
* Snow Making_ac
* vertical_drop

# [Modeling](https://github.com/RDallavia/Springboard/blob/main/ProductPricingModel/Notebooks/05_modeling.ipynb)
we now take our model for ski resort ticket price and leverage it to gain some insights into what price Big Mountain's facilities might actually support, as well as explore the sensitivity of changes to various resort parameters. Note, this relies on the implicit assumption that all other resorts are largely setting prices based on how much people value certain facilities. This means comparable prices are correct.

We can now use our model to gain insight into what Big Mountain's ideal ticket price could/should be, and how that might change under various scenarios.

What data should we include in our model? The motivation for this entire project was based on the sense that Big Mountain needs to adjust its pricing. One way to phrase this problem: We want to train a model to predict Big Mountain's ticket price based on data from all other resorts! We don't want Big Mountain's current price to bias this. We want to calculate a price based only on its competitors. So, Big Mountain is excluded from our design matrix.

```Python
# Remove Big Mountain from model design matrix and target
X = ski_data.loc[ski_data.Name != "Big Mountain Resort", model.X_columns]
y = ski_data.loc[ski_data.Name != "Big Mountain Resort", 'AdultWeekend']

# Check vector lengths
len(X), len(y)
(276, 276)

# Apply our model to our design matrix of features and response vector
model.fit(X, y)
Pipeline(steps=[('simpleimputer', SimpleImputer()), ('standardscaler', None),
                ('randomforestregressor',
                 RandomForestRegressor(n_estimators=20, random_state=47))])

# Cross-validate the model
cv_results = cross_validate(model, X, y, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)

# Grab our 5 fold cv scores
cv_results['test_score']
array([-11.65509821,  -9.06544545, -11.51256364,  -8.23481818,
       -10.73297273])

# View mean absolute error stats
mae_mean, mae_std = np.mean(-1 * cv_results['test_score']), np.std(-1 * cv_results['test_score'])

mae_mean, mae_std
(10.240179642857143, 1.361269673309552)
```
These numbers will inevitably be different to those in the previous step that used a different training data set. They should, however, be consistent. It's important to appreciate that estimates of model performance are subject to the noise and uncertainty of data!

With these numbers in hand, we can generate a preliminary ticket price for Big Mountain.

```python
# Run Big Mountain data
X_bm = ski_data.loc[ski_data.Name == "Big Mountain Resort", model.X_columns]
y_bm = ski_data.loc[ski_data.Name == "Big Mountain Resort", 'AdultWeekend']

# Obtain predictions
bm_pred = model.predict(X_bm).item()
```
Big Mountain Resort modelled price is $93.82, actual price is $81.00.
Even with the expected mean absolute error of $10.38, this suggests there is room for an increase.

Big Mountain Resort modelled price is $108.30, actual price is $81.00.
Even with the expected mean absolute error of $10.24, this suggests there is room for an increase. A ticket increase of $9.00 would cover the cost of the new run if half of the annual 350,000 visitors bought weekend passes.

Features that came up as important in the modeling (not just our final, random forest model) included:

* vertical_drop
* Snow Making_ac
* total_chairs
* fastQuads
* Runs
* LongestRun_mi
* trams
* SkiableTerrain_ac

Big Mountain outperformed a plurality of comparable resorts on all of the above metrics except `trams`.

Management assumes the resort operates within a market where people pay more for certain facilities, and less for others. Knowing how facilities support a given ticket price is valuable business intelligence. This is where our model comes in.

The business has shortlisted some options:

1. Permanently closing down up to 10 of the least used runs. This doesn't impact any other resort statistics.

2. Increase the vertical drop by adding a run to a point 150 feet lower down but requiring the installation of an additional chair lift to bring skiers back up, without additional snow making coverage

3. Same as number 2, but adding 2 acres of snow making cover

4. Increase the longest run by 0.2 mile to boast 3.5 miles length, requiring an additional snow making coverage of 4 acres

Scenario 1: The model says closing one run makes no difference. Closing 2 and 3 successively reduces support for ticket price and so revenue. If Big Mountain closes down 3 runs, it seems they may as well close down 4 or 5, as there's no further loss in ticket price. Increasing the closures down to 6 or more leads to a large drop.

Scenario 2: In this scenario, Big Mountain is adding a run, increasing the vertical drop by 150 feet, and installing an additional chair lift. Per our model, it supports a ticket price increase of $3.00.

Scenario 3: In this scenario, we are repeating the previous one, but adding 2 acres of snow making. It too supports a ticket price increase of $3.00.

Scenario 4: This scenario calls for increasing the longest run by .2 miles and guaranteeing its snow coverage by adding 4 acres of snow making capability. It turns out that it supports no ticket price increase whatsoever.


