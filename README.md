![Header](../images/Spain_nightlight.jpg)
# Predicting Energy Markets in Spain
**Author**: Evan Holder<br>
**Flatiron Data Science**: Capstone Project<br>
[Github Link](https://github.com/EvanHolder/Predicting-Energy-Markets-in-Spain)<br>
[Presentation Link]()<br>
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
# Modeling Price
**Lasso Regression**

**XGBoost**

**One-to-One Neural Network**

**24-to-24 Neural Network**

**LSTM Neural Network**

**DNN-LSTM Neural Network**
# Modeling Price Residual

# Modeling Price Components

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


