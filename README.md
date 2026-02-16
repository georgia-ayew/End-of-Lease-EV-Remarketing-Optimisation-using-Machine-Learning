# End-of-Lease-EV-Remarketing-Optimisation-using-Machine-Learning
=========================

Built a predictive model to estimate UK vehicle resale value and optimise end-of-lease disposal decisions using vehicle age, mileage, and specification data.

# Executive Summary

From findings in our Random Forest Regression we were able to accuartely predict the listing value of cars being remarketed through autotrader. 

This allows us to give indepth statistically retrieved insights to remarketing companies on resale value. It also allowed us to gain insight on which features cause under or over valuations - this will allow companies to understand where value is lost in resale and where cars are overpriced to reduce stock hold and loss of revenue.

The final model was built on autotrader listing prices between 2017-2023 in the UK. This is a broad approach that can be tailored to the remarketing companies stock and previous sales data to create a unique soloution.

## Key Takeaways 

Vehicle value is driven primarily by vehicle condition, age, regulatory standard, and usage intensity, rather than brand or cosmetic characteristics. 

The model confirms that structural factors reflecting longevity, efficiency, and regulatory compliance are the most reliable predictors of retained value.

These insights can be used to:

- Improve vehicle valuation accuracy

- Inform residual value forecasting

- Support pricing strategy and inventory decisions

- Identify high-value vehicle segments

# In Depth Insights

## Key Drivers of Vehicle Value

A machine learning model was developed to identify the primary factors influencing vehicle value. Using SHAP (Shapley Additive Explanations), we quantified the relative importance and directional impact of each feature on predicted vehicle prices.

**Primary Value Drivers**

The analysis shows that vehicle value is predominantly influenced by emissions standard, mileage, vehicle age, and engine characteristics.

Features that most strongly increase vehicle value
Newer emissions standards (particularly Euro 5 and Euro 6) are associated with higher vehicle values, reflecting regulatory desirability and improved environmental performance.

Lower mileage vehicles consistently command higher values, indicating wear and usage remain major determinants of resale price.
Newer vehicles (lower Premium_Age) significantly increase value, reflecting depreciation over time.

Medium engine sizes (1.4–2.0L) tend to positively influence value, likely reflecting an optimal balance between performance and efficiency.

**Features that most strongly decrease vehicle value**

Higher mileage, especially when accumulated rapidly (high Mileage_Delta or Mileage_per_Year), substantially reduces value.
Older vehicles show consistent depreciation effects.

Higher ownership turnover (more previous owners or higher Owners_per_Year) slightly reduces value, reflecting perceived risk and wear.


## Business Problem 

Are we able to accurately predict listing prices of used cars in order for remarketing businesses to optimise pricing and tailor fleets to hold vehicles that fit their life span cycle whilst retaining highest value.

## Data Science Problem

Using supervised Random Forest Regression algorithms are we able to establish accurate price evaluations made up of core features of the vehicle and gain actionable insight into what contributes to value in used vehicles.



## Machine Learning Task

We will do a preliminary logistic regression to observe feature importance and effectiveness of our feature selection before using a pipeine to evaluate multiple random forest regression algorithms and select our most accurate model. Our learning will involve preparing our data for modelling, this being done through preprocessing, feature engineering and feature selection. This will be an iterative process based on evaluation metrics assessing how effectively our model is separating on the features we utilise.



## The Data 

The dataset contains data sourced from Autotrader UK a leading automotive marketplace website. There are 3,685 data points each representing a unique vehicle listing and distinct features.

Our aim is to use analysis and modelling tools to create actionable insights on feature importance in remarketing value. Allowing remarketing companies to predict vehicle sale value and decide when best to remarket vehicles to maximise return.

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

