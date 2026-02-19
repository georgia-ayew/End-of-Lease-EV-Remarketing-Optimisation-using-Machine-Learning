# End-of-Lease-EV-Remarketing-Optimisation-using-Machine-Learning

Built a predictive model to estimate UK vehicle resale value and optimise end-of-lease disposal decisions using vehicle age, mileage, and specification data.

# Executive Summary

This model provides a data-driven framework for Residual Value (RV) Management. While age and mileage are the baseline drivers of depreciation, our analysis reveals high-alpha opportunities for remarketing teams to capture hidden value by timing sales around technical and market-specific lifecycle shifts.

# Actionable Insights 

## Lifecycle Timing

The model identifies Lifecycle_Stage_Late as a high-risk indicator for rapid depreciation. When a manufacturer releases a new facelift or generation, the previous model's desirability drops instantly.

**Strategic Recommendation:** Implement an Early Exit Program. Monitor assets flagged as Late Lifecycle and offer customers incentives to upgrade 3–6 months before the new model launch.

**ROI Optimisation:** Capturing the "Current Model" premium before the market is flooded with previous-gen stock can preserve 8-12% of an asset's residual value compared to waiting for contract maturity.

## Segment-Specific Mileage Arbitrage

Depreciation is non-linear. Using Mileage_per_Year, the model identifies "Value Plateaus"—mileage brackets where the price remains stable despite continued use.

**Strategic Recommendation:** Deploy Extension Tactics for under-utilized assets. If the model identifies a plateau (e.g., between 40k and 60k miles for premium brands), these vehicles are prime candidates for lease extensions.

**ROI Optimisation:** By keeping a vehicle in service during a plateau, you maximize rental income with negligible impact on the final disposal price, significantly increasing the total lifetime ROI of the asset.

## Regional and Regulatory Reallocation

With Emission Class_Euro 6 and Fuel type acting as primary price drivers, geography becomes a lever for profit. Regulatory shifts like ULEZ create artificial "highs" and "lows" in market value.

**Strategic Recommendation:** Execute Geographic Arbitrage. Use the model to identify assets that are "High Risk" in urban zones (Euro 5 or below) and move them to rural auction hubs.

**ROI Optimisation:** Reallocating Euro 6 and Hybrid stock to urban retail-ready channels captures a "Compliance Premium," while moving older stock to less regulated areas ensures faster stock-turn and prevents aged-inventory write-downs.

## Spec-Driven Channel Selection

SHAP analysis reveals that features like Gearbox_Manual or certain Body types (e.g., MPVs) carry a heavy "liquidity penalty," especially in the premium segment.

**Strategic Recommendation:** Automate Tiered Disposal Channels. Use the model's predicted margin to automatically route stock:

- High-Spec/Auto/SUV: Direct to B2C (Retail) platforms to capture maximum margin.

- Low-Spec/Manual/Hatchback: Direct to B2B (Wholesale) auctions to minimize holding costs.

**ROI Optimisation:** By identifying "Liquidity Traps" early, you avoid the high holding costs of cars that sit on retail lots for 60+ days, ensuring capital is recycled back into higher-performing assets.

## Remarketing Risk Score Formula 

**RiskScore=0.4(Lifecycle)+0.3(DepreciationPhase)+0.2(LiquidityFactor)+0.1(UsageBias)**

This allows fleet managers to better understand wether assets are stable or require remarketing to prevent profit loss on a scale of 1-10.

**8-10 Critical Exit:** Sell within 14 days. Asset is likely in "Late Lifecycle" and losing value rapidly. Use wholesale auctions for speed.

**5-7 Retail Target:** Active Marketing. High-demand specs (e.g., SUVs) that are entering Mid-Lifecycle. Use B2C channels to capture retail margin.

**3-4 Hold/Extend:** Maximize Utility. Asset is in a "Value Plateau." Can be kept in service or leased longer with minimal ROI impact.

**1-2 Prime Stock:** Premium Hold. Likely a new-model Hybrid/SUV with high brand strength. These are "Safe Havens" for capital.



## Business Problem 

Are we able to accurately predict listing prices of used cars in order for remarketing businesses to optimise pricing and tailor fleets to hold vehicles that fit their life span cycle whilst retaining highest value.

## Data Science Problem

Using supervised Random Forest Regression algorithms are we able to establish accurate price evaluations made up of core features of the vehicle and gain actionable insight into what contributes to value in used vehicles.



## Machine Learning Task

We will do a preliminary logistic regression to observe feature importance and effectiveness of our feature selection before using a pipeine to evaluate multiple random forest regression algorithms and select our most accurate model. Our learning will involve preparing our data for modelling, this being done through preprocessing, feature engineering and feature selection. This will be an iterative process based on evaluation metrics assessing how effectively our model is separating on the features we utilise.



# The Data 

The dataset contains data sourced from Autotrader UK a leading automotive marketplace website. There are 3,685 data points each representing a unique vehicle listing and distinct features.

Our aim is to use analysis and modelling tools to create actionable insights on feature importance in remarketing value. Allowing remarketing companies to predict vehicle sale value and decide when best to remarket vehicles to maximise return.

## Data Dictionary

<h3>Primary Asset Drivers</h3>
<table style="width:100%; border: 1px solid black; border-collapse: collapse;">
  <tr style="background-color: #f2f2f2;">
    <th style="border: 1px solid black; padding: 8px;">Feature</th>
    <th style="border: 1px solid black; padding: 8px;">Type</th>
    <th style="border: 1px solid black; padding: 8px;">Definition</th>
    <th style="border: 1px solid black; padding: 8px;">Remarketing Strategy</th>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 8px;"><b>Car_Age</b></td>
    <td style="border: 1px solid black; padding: 8px;">Float</td>
    <td style="border: 1px solid black; padding: 8px;">Years since registration</td>
    <td style="border: 1px solid black; padding: 8px;">Baseline depreciation anchor</td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 8px;"><b>Mileage_per_Year</b></td>
    <td style="border: 1px solid black; padding: 8px;">Float</td>
    <td style="border: 1px solid black; padding: 8px;">Total Miles / Age</td>
    <td style="border: 1px solid black; padding: 8px;">Detects high-intensity usage</td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 8px;"><b>Remaining_Life</b></td>
    <td style="border: 1px solid black; padding: 8px;">Float</td>
    <td style="border: 1px solid black; padding: 8px;">Est. useful life remaining</td>
    <td style="border: 1px solid black; padding: 8px;">Retail-Ready vs Auction-Only</td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 8px;"><b>Brand_Strength</b></td>
    <td style="border: 1px solid black; padding: 8px;">Float</td>
    <td style="border: 1px solid black; padding: 8px;">Brand prestige score</td>
    <td style="border: 1px solid black; padding: 8px;">Quantifies the price floor</td>
  </tr>
</table>

<h3>Advanced ROI Levers</h3>
<table style="width:100%; border: 1px solid black; border-collapse: collapse;">
  <tr style="background-color: #f2f2f2;">
    <th style="border: 1px solid black; padding: 8px;">Feature</th>
    <th style="border: 1px solid black; padding: 8px;">Definition</th>
    <th style="border: 1px solid black; padding: 8px;">Strategic Insight</th>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 8px;"><b>Lifecycle_Stage</b></td>
    <td style="border: 1px solid black; padding: 8px;">New, Mid, Late</td>
    <td style="border: 1px solid black; padding: 8px;"><b>Late</b> indicates a model refresh; sell now to avoid price cliff.</td>
  </tr>
  <tr>
    <td style="border: 1px solid black; padding: 8px;"><b>Depreciation_Phase</b></td>
    <td style="border: 1px solid black; padding: 8px;">S-Curve Position</td>
    <td style="border: 1px solid black; padding: 8px;">Identify <b>Value Plateaus</b> where holding the asset is low-cost.</td>
  </tr>
</table>

# Methodology 

**Notes on Methodology**

We assigned Log_Price as our target and performed supervised learning throughout the research. Evaluation Metrics aligned with best practise for regression models and was consistent throughout. By evaluating the value of our residuals we were able to observe accuracy and learn from the discrepancies.

### Data Cleaning 

Our dataset had very little need for cleaning. We had missing values that were either dropped or imputed.

### EDA 

We found that the listings showed cars registered between 1992-2023. The price average was £5789 and there was a large variance in price typically ranging between ~ £3,250 and £8,250.

![alt text](image-3.png)

We found that there was a negative correlation between price and mileage and price and age after removing illogical outliers.

![alt text](image-4.png)

![alt text](image-5.png)


When isolating engine size bins against price we found the correlation followed a simliar negative distrtibution for each engine type for small-large with performance engines having less of a trend and greater outliers.

This allowed us to determine that outliers would most likely fall into this engine size category and could be related to specific highly desired vehicles. However we saw the greatest variance in price in the medium sized engine with the greatest outliers but also being the most common engine size in our data.

![alt text](image-6.png)

## Feature Engineering 

When creating our new features we added depreciation features such as; mileage per year and age bands. We also added vehicle usage intensity, expected vs actual mileage and owners per year.

A log transformation was applied to our right skewed features (mileage and age).

Other features included model oriented features such as Premium brand indicator, Model Name and Brand Name. 

Obeserving Brand vs price we found the highest brand mean values were in BMW, Audi, Nissan, Mercedes and Volkswagen with Mercedes having the widest distribution.

![alt text](image-7.png)

Vehicle size features such as a door category and family suitable car were added for insights into practicality driving prices.

Additional features included; engine per seat ratio,service history age relationship and premium age interaction.

## Preprocessing 

In preparation we first split our target variable from our other features. We then removed any features we added for analysis that were indicative of price. 

To ensure our model was interpretable after exploring multiple features in our logistic regression we kept key features that company can observe;

**Numeric / continuous**

    'Car_Age',
    'Mileage_per_Year',
    'Engine',
    'Brand_Strength',       
    'Remaining_Life',      
    
    
    
**Categorical**

    'Fuel type',
    'Body type',
    'Gearbox',
    'Emission Class',
    'Brand',
    'Lifecycle_Stage',
    'Depreciation_Phase',
    
**Target**

    'Log_Price'

## Random Forest Regression 

We used a pipeline to transform test data and explored parameters. Our testing concluded;

Fitting 5 folds for each of 50 candidates, totalling 250 fits

**Best params:** 
**'regressor__n_estimators':** 800

**regressor__min_samples_split:** 2

**regressor__min_samples_leaf:** 1 

**regressor__max_features:** 0.7 

**regressor__max_depth:** 20

**Train R²:** 0.9853116675342399

**Test R²:** 0.8739340341865218

**RMSE (£):** 1340.0137857252557

**MAE (£):** 770.2602978678395

**R²:** 0.9156614907016334



# Model Evaluation 

## Residual Analysis

![alt text](image-8.png)

**Log-space residuals:** The residuals are fairly evenly scattered around zero with no obvious trend, indicating that the model fits the log-transformed target reasonably well. Most predictions are close to the true log-values, and there are no extreme systematic biases.

**Actual-price residuals:** The residuals in dollars show more spread, especially for higher-priced cars. This is expected because the model was trained on log-transformed prices — errors in log-space translate to larger dollar errors for more expensive cars. Some outliers exist, likely representing high-priced vehicles where the model underestimates or overestimates slightly.

**Interpretation:**
Overall, the model is capturing the main patterns of car pricing well.
Large deviations for expensive cars are a known effect of log-transforming targets, and could be reduced with more data for high-priced cars or specialized models.

## Predictive Accuracy

The combination of SHAP plots and residuals suggests good predictive performance:

- Most predictions lie close to the 45° line in actual vs predicted plots.
- No strong heteroscedasticity is visible in log-space, which validates the choice of log transformation.

## Feature Insights (SHAP Analysis)

### SHAP Summary Plot (impact on output)

![alt text](image-9.png)

Top positive/negative drivers of price:
Car_Age: The most important feature; newer cars (low age) push predicted prices higher, older cars lower.
Lifecycle Stage: Late-stage cars (Lifecycle_Stage_Late_True/False) significantly affect price — newer lifecycle stages increase value, later stages decrease it.
Mileage_per_Year: Higher mileage reduces price; lower mileage increases price.
Brand_Strength: Stronger brands positively influence car value.
Engine: Larger engine size slightly increases price.

### SHAP Feature Importance (bar plot)

![alt text](image-10.png)

**Confirms the ranking of features:**

- Car_Age — dominates impact on price.
- Lifecycle Stage — early vs late stage has strong effect.
- Mileage_per_Year — a key negative driver.
- Brand_Strength — differentiates premium from standard cars.
- Engine and other categorical features — moderate impact.

**Interpretation:**

- The model aligns with real-world expectations: newer cars with strong brands, lower mileage, and better lifecycle stage command higher prices.
- Categorical features like gearbox type and emission class have minor effects but still contribute to predictions.

## Conclusion

**Performance:**

- Model fits log-prices well, translating into reasonable predictions in actual prices.
- Residuals show good randomness around zero, suggesting no major systematic bias.

**Drivers of car price:**

- Age, lifecycle stage, mileage, and brand strength are the strongest determinants.
- Engine size and gearbox type have secondary influence.

**Potential Improvements:**
- For expensive cars, consider additional features or specialized models to reduce large residuals.
Explore interaction terms (e.g., Age × Mileage) for subtle patterns.




































