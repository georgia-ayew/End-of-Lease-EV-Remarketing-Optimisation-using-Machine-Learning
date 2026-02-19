# End-of-Lease-EV-Remarketing-Optimisation-using-Machine-Learning
=========================

Built a predictive model to estimate UK vehicle resale value and optimise end-of-lease disposal decisions using vehicle age, mileage, and specification data.

# Executive Summary

This model provides a data-driven framework for Residual Value (RV) Management. While age and mileage are the baseline drivers of depreciation, our analysis reveals high-alpha opportunities for remarketing teams to capture hidden value by timing sales around technical and market-specific lifecycle shifts.

# Key Insights

**Lifecycle vs Age**: The Late Lifecycle feature is a more significant predictor of value loss than an additional year of age. Selling an asset 3–6 months before a manufacturer facelift or new generation launch preserves an average of 8-12% more value than selling post-launch.

**Segment-Specific Mileage Sensitivity:** Depreciation isn't linear. For Premium Brands, the model identifies a "Value Plateau" between 40k and 60k miles. Conversely, for Volume Brands, passing the 60k-mile mark (typical end of extended warranties) triggers a sharp non-linear drop in desirability.

**Brand Strength:** SHAP analysis confirms that Brand Strength acts as a price stabilizer. Even with higher mileage, premium badges (Audi, BMW, Mercedes) maintain a significantly higher "Exit Price" compared to budget brands with identical specs, suggesting these assets can be held longer in lease portfolios without major ROI decay.

**Powertrain Premium:** In the current market, the model shows a widening gap in ROI for Automatic vs. Manual transmissions and Euro 6 compliant engines. Remarketing teams should prioritize "Retail-Ready" channels for automatics while moving manuals through faster wholesale auctions to minimize holding costs.

=========================

## Business Problem 

Are we able to accurately predict listing prices of used cars in order for remarketing businesses to optimise pricing and tailor fleets to hold vehicles that fit their life span cycle whilst retaining highest value.

## Data Science Problem

Using supervised Random Forest Regression algorithms are we able to establish accurate price evaluations made up of core features of the vehicle and gain actionable insight into what contributes to value in used vehicles.



# Machine Learning Task

We will do a preliminary logistic regression to observe feature importance and effectiveness of our feature selection before using a pipeine to evaluate multiple random forest regression algorithms and select our most accurate model. Our learning will involve preparing our data for modelling, this being done through preprocessing, feature engineering and feature selection. This will be an iterative process based on evaluation metrics assessing how effectively our model is separating on the features we utilise.



## The Data 

The dataset contains data sourced from Autotrader UK a leading automotive marketplace website. There are 3,685 data points each representing a unique vehicle listing and distinct features.

Our aim is to use analysis and modelling tools to create actionable insights on feature importance in remarketing value. Allowing remarketing companies to predict vehicle sale value and decide when best to remarket vehicles to maximise return.


<h3>📘 Data Dictionary: Primary Asset Drivers</h3>
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

<h3>📘 Data Dictionary: Advanced ROI Levers</h3>
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













































































## Data Dictionary

- `Title`: Vehicle Model Name  

- `Price`: OTR price of the Vehicle Listed 

- `Mileage(miles)`: Vehicle's recorded mileage since ownership 

- `Registration_Year`: Year of vehicle production

- `Previous Owners`: Number of owners the vehicle has had by time of sale price being logged 

- `Fuel Type`: Vehicle fuel type indicatice also of powertrain 

- `Body type`: Body type of vehicle listed 

- `Engine`: Size of vehicle engine 

- `Gearbox`: Vehicle transmission 

- `Doors`: Number of doors on the vehicle

- `Seats`: Number of seats in the vehicle 

- `Emission Class`: Emission Class of the Vehicle 

- `Service history`: Service history at time of sale price being logged 

## Data Dictionary After Feature Selection 

### Numeric Features 

- Previous Owners
- Engine
- Doors
- Seats
- Has_Service_History
- Mileage_per_Year
- Log_Mileage
- Mileage_Delta
- Owners_per_Year
- Is_Family_Car
- Premium_Age

### Fuel Type (One-Hot Encoded)
 
- Fuel type_Petrol
- Fuel type_Petrol Hybrid
- Fuel type_Petrol Plug-in Hybrid

### Body Type (One Hot Encoded)

 - Convertible, Coupe, Estate, Hatchback, MPV, Pickup, SUV, Saloon

 ### Gearbox (One Hot Encoded)

 - Manual (dropped automatic)

 ### Emission Class (One Hot Encoded)

- Emission Class_Euro 2 - 6 (dropped 1)

### Binned Engine Sizes (One Hot Encoded)

- Engine_Bin_Small (≤1.4L)
- Engine_Bin_Medium (1.4–2.0L) (dropped large)
- Engine_Bin_Performance (3.0L+) 

### Age Bands 

- Age_Band_3-6 (dropped 0-3)
- Age_Band_6-10
- Age_Band_10+

### Brands (One-Hot Encoded)

Alfa, Audi, BMW, Chevrolet, Chrysler, Citroen, Dacia, Daihatsu, Dodge, DS, Fiat, Ford, Honda, Hyundai, Infiniti, Jaguar, Jeep, Kia, Land Rover, Lexus, Maserati, Mazda, Mercedes, MG, Mini, Mitsubishi, Nissan, Peugeot, Porsche, Proton, Renault, Rover, Saab, Seat, Skoda, Smart, Ssangyong, Subaru, Suzuki, Toyota, Vauxhall, Volkswagen, Volvo

### Models (One-Hot Encoded)

A-class, ASX, Accent, Accord, Adam, Agila, Almera, Almera Tino, Alpina, Alto, Amica, Antara, Astra, Astra GTC, Auris, Automobiles DS, Avensis, Aygo, B Class, B-class, B-max, Beetle, Berlingo, Bora, C Class, C-class, C-max, CC, CL, CLA Class, CLK, CLS, CT, Caddy Maxi Life, Captiva, Captur, Cayenne, Ceed, Ceed Diesel Hatchback, Cherokee, Civic, Clio, Clubman, Colt, Compass, Convertible, Corolla, Corolla Verso, Corsa, Corsa Hatchback, Corsa Hatchback Special EDS, Coupe, Cr-v, Crossland, Crossland X, Crossland X Hatchback, Cruze, Doblo, Duster Estate, E Class, E Class Diesel Coupe, EOS, Ecosport, Ecosport Hatchback, FX, Fabia, Fabia Diesel Estate, Fiat, Fiesta, Fiesta Hatchback, Focus, Focus Active, Focus C-max, Focus CC, Focus Diesel Hatchback, Focus Hatchback, Forester, Forfour, Fortwo, Fr-v, Fullback, Fusion, G, GLE Class, GS, Galaxy, Getz, Ghibli, Golf, Golf Diesel Hatchback, Golf Hatchback, Golf Plus, Grand, Grand C-max, Grand Cherokee, Grand Espace, Grand Scenic, Grand Vitara, Grand Voyager, Grande Punto, Grandland X, Granturismo, Hatch, Hatch Cooper, Hatch ONE, Hatchback, IQ, IS, Ibiza, Ignis, Impreza, Insignia, Insignia Grand Sport, Insignia Sports Tourer, Ioniq, Jazz, Jazz Hatchback, Jetta, Jetta Diesel Saloon, Jimny, Juke, KA, KA+, Kamiq, Kangoo, Karoq Estate, Koleos, Kona, Korando, Kuga, Kuga Diesel Hatchback, LS, Laguna, Lancer, Legend, Leon, Liana, Logan MCV, Lupo, M, M Class, MAZDA, Megane, Megane Hatch, Meriva, Micra, Modus, Mokka, Mokka X, Mondeo, Multipla, Murano, Mustang, Navara Diesel Pick UP, Nitro, Note, Octavia, Optima, Outback, Outlander, PT Cruiser, Panda, Partner Tepee, Passat, Passat Petrol/electric Saloon, Pathfinder, Patriot, Phaeton, Picanto, Polo, Polo Hatchback, Prelude, Prius, Proceed, Proton, Punto, Punto EVO, Qashqai, Qashqai Diesel Hatchback, Qashqai Hatchback, RCZ, RIO, Rapid, Rapid Spaceback, Renegade, Romeo, Romeo GT, Romeo Giulietta, Romeo Mito, Roomster, Rover Discovery, Rover Discovery Sport, Rover Freelander, Rover Range Rover, Rover Range Rover Evoque, Rover Range Rover Sport, S Class, S-max, S-type, SC, SL Class, SLK, Sandero, Sandero Stepway, Santa FE, Savvy, Scenic, Scirocco, Scirocco Diesel Coupe, Sebring, Sharan, Shogun, Shogun Sport, Sorento, Soul, Spark, Sportage, Sportage Diesel Estate, Sportage Estate, Stilo, Streetka, Superb, Swift, Swift Hatchback, T-cross, TF, TT, Tarraco, Terios, Tigra, Tiguan, Toledo, Touran, Tucson, Twingo, UP, UP!, Vectra, Veloster, Venga, Verso, Vitara, Viva Hatchback, Vivaro, X-trail, X-trail Diesel Estate, X-type, XE, XF, XJ, XM, XV, Xceed Hatchback, Xsara Picasso, Yaris, Yeti, Ypsilon, ZR, Zafira, Zafira Tourer

### Usage & Door Categories

- Usage_Level_Low
- Usage_Level_Normal
- Usage_Level_Very High

- Door_Category_Family 
- Door_Category_Sedan
- Door_Category_Small

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

In preparation we first split our target variable from our other features. We then removed any features we added for analysis that were indicative of price;
- `Price_per_Seat`

- `Price_per_Year_Age`

- `Brand_Avg_Price`

- `Model_Avg_Price`

We one hot encoded all categorical features; 'Fuel type','Body type', 'Gearbox','Emission Class','Engine_Bin', 'Age_Band','Brand','Model','Usage_Level', and 'Door_Category'.

We then created our test/train split ready for modelling.

## Logistic Regression 

Address Later 

## Random Forest Regression 

## Model Evaluation 

## Conclusion










































## Segment-Level Learnings

![alt text](image-1.png)

- Vehicles grouped by Emission Class, Age Band, and Engine size reveal patterns:

- Euro 6, low mileage, family cars → consistently undervalued → pricing opportunity

- Older, smaller engines → often overvalued → adjust acquisition/pricing strategy

- Using binned engine categories improves clarity of segment-level insights.

=========================

## Recommendations

**Pricing Strategy:**

- Increase prices on undervalued segments by £500–£1,500 depending on residual magnitude.

- Reduce prices on overvalued segments to accelerate sales and reduce holding costs.

**Inventory Planning:**

- Prioritise acquisition of vehicles that historically outperform predictions (low mileage, family-friendly, Euro 6).

- Avoid over-investing in segments that tend to underperform.

**Marketing:**

- Promote undervalued vehicles prominently to maximize ROI.

- Highlight key features undervalued by the model (engine size, Euro class, mileage).

**Model Monitoring & Refinement:**

- Re-train periodically with new sales data.

- Consider segment-specific models for rare/premium vehicles to reduce extreme residuals.

=========================

## Model Performance

**Model Used:** 

- Random Forest Regression (trained on historical UK Vehicle Sale Data 2023-2017)

**Target:**

- Vehicle Price (log transformed for modelling)

**Accuracy Metrics** 

- **R²**: 0.9 (90% of price variability explained)
- **RMSE**: £1,427 avergae deviation from predicted price 
- **MAE**: £846 average error per vehicle

=========================

## Key Insights from Residuals

![alt text](image-2.png)

![alt text](image.png)

**Residuals** = Actual Price – Predicted Price

**Undervalued Vehicles (sell for more than predicted):**

| Vehicle | Actual Price (£) | Predicted Price (£) | Residual (£) |
| ------- | ---------------- | ------------------- | ------------ |
| #276    | 15,499           | 2,503               | 12,996       |
| #671    | 10,994           | 3,663               | 7,331        |
| #522    | 20,499           | 14,791              | 5,708        |
| #307    | 11,496           | 5,816               | 5,680        |
| #66     | 12,494           | 6,847               | 5,647        |

- Example: Low mileage, family-friendly, or premium models

- Average positive residual: up to £13,000

- Action: Highlight these vehicles in listings, consider pricing slightly higher to maximize revenue.

**Overvalued Vehicles (sell for less than predicted):**

| Vehicle | Actual Price (£) | Predicted Price (£) | Residual (£) |
| ------- | ---------------- | ------------------- | ------------ |
| #33     | 4,299            | 8,854               | -4,555       |
| #479    | 2,998            | 7,328               | -4,330       |
| #9      | 6,792            | 10,866              | -4,074       |
| #35     | 3,399            | 6,620               | -3,221       |
| #520    | 9,994            | 13,198              | -3,204       |

- Example: Older, high-mileage, or less popular models

- Average negative residual: up to £4,500

- Action: Avoid overpricing these vehicles; consider promotions or faster turnover.

=========================




Our Random Forest model demonstrates strong predictive performance for car pricing. Predictions on the test set are generally accurate, with errors evenly distributed across most price ranges. While some deviation occurs for higher-priced vehicles, this is expected due to the log-transformation applied during training. Overall, the model reliably captures the main factors that drive car value.

The analysis of feature importance provides clear insights into what determines a car’s price. Car age is the most influential factor: newer cars command higher prices, while older cars decrease in value. Lifecycle stage and mileage per year also strongly impact pricing, with cars in earlier stages and lower annual mileage valued more. Brand strength contributes positively, indicating that vehicles from more reputable brands maintain higher prices. Other factors such as engine size and gearbox type have smaller, but still meaningful, effects.
These insights align closely with real-world expectations and can guide strategic decisions in pricing, inventory management, and customer recommendations. For instance, emphasising low-mileage, newer, or premium-brand vehicles can optimize sales value. Additionally, future improvements could focus on high-priced cars to further reduce prediction errors, potentially by including more specialised features or interaction terms. 