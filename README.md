<table><tr>
<td><img src="https://github.com/EvanHolder/Predicting-Energy-Markets-in-Spain/blob/main/images/solar_panels.jpg" style="width:320px;height:250px"/></td>
<td><img src= "https://github.com/EvanHolder/Predicting-Energy-Markets-in-Spain/blob/main/images/spain_grid.gif" style="width:320px;height:250px"/></td>
<td><img src= "https://github.com/EvanHolder/Predicting-Energy-Markets-in-Spain/blob/main/images/wind_turbine.jpg" style="width:320px;height:250px"/></td> 
</tr></table>

# Predicting Energy Markets in Spain
**Author**: Evan Holder<br>
**Flatiron Data Science**: Capstone Project<br>
[Github Link](https://github.com/EvanHolder/Predicting-Energy-Markets-in-Spain)<br>
[Presentation Link]()<br>

# Repo Contents
* README.MD
* Images
* Notebooks
    * LassoRegression.ipynb
    * NeuralNets.ipynb
    * RandomForest.ipynb
    * XGBoost.ipynb
    * capstone.ipynb
    * energy_visuals_exploration.ipynb
* scripts
    * EDA_cleaning_components.py
    * functions.py
    * weather_scraper.py



# Background
With the rise renewable energies, which are significantly impacted by the weather, it has become harder to balance energy supply and demand and keep the grid stable.  Grid instability often results in price uncertainty. By more accurately modeling the price of electricity, we can minimize the effects of a rapidly changing supply and demand balance. Or in other words, the better NEMOs can model the price of electricity, the better they can regulate energy production, limit wasted resources, and more efficiently deliver electricity to the consumer.

**Can we use information about energy generation, transmission, the weather, and the day ahead price to accurately predict the price of electricity tomorrow?**  

# Business Problem
With the rise renewable energies, which are significantly impacted by the weather, it has become harder to balance energy supply and demand and keep the grid stable.  Grid instability often results in price uncertainty. By more accurately modeling the price of electricity, we can minimize the effects of a rapidly changing supply and demand balance. Or in other words, the better TSO's can model the price of electricity, the better they can regulate energy production, limit wasted resources, and more efficiently deliver electricity to the consumer.
# Data
The data for this project focuses on the electricity market in Spain. Generation, transmission, and load data were sourced from the [entso-e Transparency Platform](https://transparency.entsoe.eu/dashboard/show). Weather data was scraped from [wunderground.com](wunderground.com). All electrical price data was retrieved from the Spanish TSO [Red Electric Espana](https://www.esios.ree.es/en/market-and-prices). 
* Training set: 2015-2019
* Validation set: 2020
* Testing set (holdout): 2021
# Methods
### Forecasting The Final Price
The approach was to start with simpler, more interpretable models and proceed to some more complex neural networks.  I began modeling the price using a Lasso Regression as lasso is easily interpretable.  It also has the added bonus that it's able to handle multicolinearity as it has automatic feature selection built in.  Lasso models performed exceptionally well when the day-ahead price was included in the predictors, but was mostly noisy without it.

Next I modeled with XGBoost. With such a dominating feature as price-day ahead was in lasso regression, it was not surprising to find the same results using XGBoost.  Because XGBoost has the ability to learn non-linear relationships, I felt optimistic that it may be able to fit better on the data without the help of the day-ahead price.  However, results were similar to lasso, and even after expanding the tree depth the model did not fit without the day-ahead price

Finally, I attempted a few variations of neural network.  The intutition was that a deep neural network may be able to uncover hidden relationships between the data and the price unseen by either lasso or XGBoost.  More notebably, I set up a few sequence to sequence (24 hour to 24 hour) input/output neural networks which again did not improve on the previous models without the use of the day-ahead price.  LSTM and ensemble DNN-LSTM networks were in hopes of capturing time-depentent sequences to sequence relationships but again model performance did not improve in any significant way without the day-ahead price.

### Forecasting Price Components
As the data did not appear to contain information on the final price, I moved on to investigate if the data could predict the other 14 price components.  In this investigation, I took two approaches: 1) model the price residual (day-ahead price minus actual price) and 2) model each individual component.  While the residual price modeling was mostly just noise, I was sucesseful in model the capacity payment, PBF tech, and secondary reserve components. Using a MultiOutputRegressor from sklearn, I was able to fit both Lasso and XGBoost models with these components as targets. The predictions from these models were summed and added to the day-ahead price to obtain final results comparable to modeling the actual price. Forecasting these three price components did decrease the SMAPE on both model types.

### Final Model Evaluation
Since all the models fit have performed extremely well, I'll selected the most interpretable model: Lasso.  In particular, the Lasso model with only five features is extremely simple, cut SMAPE by about a third, and increased r-squared.  Below, is a plot of the final model's coefficients.

![Final Lasso Model](https://github.com/EvanHolder/Predicting-Energy-Markets-in-Spain/blob/main/images/lasso_feature_importance.png)

In general, it is no surprise that `price_day_ahead` dominates the model.  The other features nudge that final price up or down.  `Renewable_lag`, `waste_lag`, `oil_lag` are all generation source. The model says that as renewable generation (feature represents other renewable outside major solar and wind) and waste generation decrease, the final price increases.  As generation from oil increases, the price also increases.  Finally  as the humidity in Bilbao decreases the price tends to decrease.

For comparison, below is the final model predictions, the `price_day_ahead`, and the actual price over the course of the first week of 2021. You can see how lasso one appears as an adjusted form of the NEMO prediction, nudging the predicted price up at each time step. 

![](https://github.com/EvanHolder/Predicting-Energy-Markets-in-Spain/blob/main/images/final_predictions.png)

In addition, the below plot shows the model residuals and day-ahead price residual over the course of the first week of 2021.  As shown, the model's residuals tend to track closer to zero than do the NEMO's predictions.

!()[https://github.com/EvanHolder/Predicting-Energy-Markets-in-Spain/blob/main/images/residuals_final.png]

# Conclusion
This project set out to see if the price was predictable from generation, transmission, weather, and price day-ahead data. In the end, the price day-ahead was a strong predictor of tomorrow's price, dominating the model. However, there are other features that nudge the price either up or down. On days when renewable generation, waste generation, and the humidity in Bilbao are up the final price tends to be cheaper.  On days when energy generation from oil is up, the final price tends to be more expensive.

The other components of final price were modeled but only three could be easily predicted with the data.  These components are `price_capacity_payment`, `price_PBF_tech`, and `price_sec_reserve`.

On the whole, most of the generation weather, and transmission variables were not helpful in predicting the final energy price.

# Next Steps
Since most of the data in this project is very noisy, it would be beneficial to sit down with an expert who understands the calculation of `price_day_ahead`. Using some of the features that the NEMOs use to come up with this price would be useful in machine learning models. 

In combination with the appropriate data a GRU neural network ensemble, and convolutional neural network (CNNs) ensemble.  GRU NNs are similar to LSTM network, but more computationally more efficient. CNN are typically used for image classification but could be adapted for multi-step price forecasting.

# For More Information
See the full analysis in the [Jupyter Notebook](https://github.com/EvanHolder/Predicting-Energy-Markets-in-Spain).

For additional info, contact Evan Holder at holderevane@gmail.com

# Repository Structure


