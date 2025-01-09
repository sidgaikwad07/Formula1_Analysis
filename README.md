# Formula1_Analysis Overview
## The project is focused on studying Formula 1 datasets for insights, analysis, and machine learning modeling of key metric predictions, such as the best pit stop strategy, drivers' championship, and race results. The key questions are answered in the project through EDA and predictive modeling, related to team/driver performance, circuit characteristics, and strategy optimization.

# Table of Contents

## 1) Introduction
### 1.1) Overview of the Project
### 1.2) Objective and Scope

## 2) Getting Started
### 2.1) Prerequisites

## 3) Features: Summary of key analyses and predictions:
###  3.1) Analysis of Circuits with longest and shortest lap times.
###  3.2) FIA championship rivalry of the most heated season that being 2021.
### 3.3) Treemap telling us the distribution of drivers nationality
### 3.4) FIA driver championship rivalry of the most heated season 2021.
### 3.5) FIA championship analysis.
### 3.6) The analysis of pit stop data.
### 3.7) The analysis of points scored by the teams and drivers.   
### 3.8) Which driver could convert from pole to win.
### 3.9) The analysis of race statistics and fastest lap records.
### 3.10) Pit_stop_strategy_optimization which feature is most important.
### 3.11) Prediction for predicts whether a driver will finish in the top 5 based on qualifying results, driver/team form, and track characteristics. 

## 4) Data
### 4.1) Datasets Used:
    4.1.1) Description of Datasets
    4.1.2) Source of Data
### 4.2) Pre-Processing Steps

## 5) Methodologies
### 5.1) Analytical approaches
### 5.2) Predictive modeling techniques
### 5.3) Tools and frameworks used

## 6) Results
### 6.1) Insights Driven from Analytics
### 6.2) Key Outcomes from Predictive Modeling

## 7) Code Structure: Explanation of Folder & File Structure
### 7.1) Analysis Scripts

## 8) Requirements
## 9) Acknowledgements
## 10) Conslusion
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### 1) Introduction
### 1.1) Overview of the Project
#### The project is a comprehensive data analysis and predictive modeling framework to gain insights into Formula 1 racing. This project makes use of historical race data to explore important performance metrics, uncover trends, and build predictive models that extend understanding from multiple perspectives of the sport. From driver and team performance analysis to race outcome and championship predictions, this project covers it all in the world of Formula 1 analytics.

### 1.2) Objective & Scope
#### The primary objective of this project is to utilize advanced data analysis techniques and machine learning models to extract meaningful insights and make accurate predictions in the context of Formula 1 racing. The project addresses the following key goals:

    1) Analytical Insights
      a) Analyze historical race data for trends in driver and team performance.
      b) Determine the characteristics of tracks that affect lap time and pit stop time.
      c) Understand drivers' and constructors' performances, influencing a race or championship outcome.

    2) Predictive Modeling
      a) Predict the optimal pit stop strategy for maximum performance.
      b) Forecast the Driver's Champion after any given season has a few early-season races.
      c) Find models with different features to predict lap times and race outcomes.

    3) Visualisation & Communication
      a) Effectively communicate data-driven insights using intuitive visualizations. 
      b) Present findings in a way to aid stakeholders in making informed decisions.
#### The scope of the project goes beyond basic analytics by introducing advanced machine learning techniques and frameworks that will create actionable insights and predictions. It is designed to serve enthusiasts, analysts, and professionals alike within the motorsport industry, providing them with a robust and versatile analysis toolkit.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### 2) Getting Started
### 2.1) Prerequisites 
#### 1) Docker
      1.1) Install Docker (Version 20.10 or higher).
      1.2) Ensure Docker Desktop or equivalent is running on your system.
#### 2) Python
      2.1) Optional: For additional customization outside the Docker environment, ensure Python (Version 3.8 or higher) is installed locally.
#### 3) Jupyter Notebook
      3.1) Jupyter Notebook is hosted via Docker, so no local installation is required.
#### 4) Required Datasets
      4.1) Place all required datasets (drivers.csv, races_cleaned_1.csv, results.csv, constructors.csv, etc.) in the F1_dataset directory, as outlined in the project structure.
#### 5) Git
      5.1) Ensure Git is installed for cloning the repository.
#### 6) Project Repository
      6.1) Clone the project repository:
            git clone https://github.com/Formula1_Analysis.git
            cd Formula1_Analysis
            
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### 3) Features: Summary of Key Analyses and Predictions
### 3.1) Analysis of Circuits with Longest and Shortest Lap Times
#### a) Objective: Identify which circuits have the longest and shortest average lap times.
#### b) Conclusion: Historically, it was the Tucson Grand Prix that boasted the longest lap times due to its extensive layout. However, the Tucson Grand Prix and the Korean Grand Prix are no longer part of the calendar. Among those circuits in the current calendar, the Belgian Grand Prix at Spa-Francorchamps has the longest lap times due to its expansive layout and high-altitude changes. The Monaco Grand Prix has the shortest lap times recorded because of its compact and narrow design. 
#### c) Visualisation: Scatter plot comparing average lap times across circuits.
![Scatter Plot for Average Lap Time Analysis](https://drive.google.com/uc?id=1jaDTr_B5g782PpLiXaDv_DJIpSjx-jkr)
![Table for Average Lap Time Analysis](https://drive.google.com/uc?id=1TKdzHaMeQns8mAe8appXAcjoK6WyvAAT)

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### 3.2) FIA Constructor Championship Rivalry: Most Heated Season 2021
#### a) Objective: Analyze the driver dynamics and rivalries within the 2021 season.
#### b) Conclusion:  The data reaffirmed the dominant performance of Mercedes and Red Bull drivers, with close battles in qualifying, race positions, and fastest laps.
#### c) Visualisation: Bar-chart for the analysis of points, and the teams.
![Most Heated Season 2021 Constructor Rivalry](https://drive.google.com/uc?id=1whmMgN7bxYe5YeEv2des7YfnmqIqdwjY)

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### 3.3) FIA Driver Championship Rivalry: Most Heated Season 2021
#### a) Objective: Explore the intense rivalry between Max Verstappen and Lewis Hamilton during the 2021 season.
#### b) Conclusion: The 2021 Formula 1 season was marked by the intense rivalry between Max Verstappen and Lewis Hamilton, with the two drivers reaching the last round of the season at the Abu Dhabi Grand Prix on 396.5 points apiece, one of the closest tussles in F1 history. In the dramatic and controversial tail-end of such a fierce season, Verstappen managed to outpace his opponent for the championship. While Valtorri Bottas, Sergio Pérez, and Carlos Sainz had competitive seasons, the season once again coalesced around Verstappen and Hamilton because their respective skills and determination made for an exciting close of the very last lap of this season.
#### c) Visualisation: Bar-chart for the analysis of points and driver names.
![Driver Rivalry 2021: Max Verstappen vs Lewis Hamilton](https://drive.google.com/uc?id=1R8vLDJLnwm0L6rWohzBQaKmDlvb9R7a5)

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### 3.4) Treemap of Drivers Nationality
#### a) Objective: Analyze the nationality distribution of Formula 1 drivers.
#### b) Conclusion: A majority of drivers hail from European nations, with the United Kingdom, Germany, and Italy leading the representation. Emerging talent from countries like Japan and Australia highlights the growing global reach of Formula 1.
#### c) Visualisation: Treemap for categorizing the nationality of F1 drivers.
![Tree Map for Nationality of Drivers](https://drive.google.com/uc?id=1FG5D4r938xtFE3YT4B2ugaVubJWGdh8g)

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### 3.5) FIA Championship Analysis
#### a) Objective: Provide a comprehensive review of past FIA championships, highlighting trends in driver and constructor dominance.
#### b) Conclusion: The bar chart illustrates the number of FIA championships won by various racing teams, highlighting the dominance of a few key teams in motorsport history. Ferrari stands out as the most successful team with over 200 championships, followed by McLaren, Mercedes, and Red Bull, each with significant contributions, though notably fewer than Ferrari. Williams and Renault also show strong performances, reflecting their historical success. In contrast, a long tail of teams like Lotus, Tyrrell, and Brawn display moderate achievements, while many other teams have won fewer championships, emphasizing the disparity in success across teams in FIA competitions. This chart underscores the legacy of a few elite teams that have consistently excelled in the sport.
#### c) Visualisation : Bar-Chart for all the previous years of analysis.
## FIA Championships Image
![FIA Championships](https://drive.google.com/uc?id=1ZR09SConf-6LVdvl14zYgFm1eZ5kJ-Ey)

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### 3.6) Analysis of Pit Stop Data
#### a) Objective: Evaluate team performance in pit stops and its influence on race outcomes.
#### b) Conclusion: From 2012 to 2024, the heatmap reveals significant advancements in pit stop efficiency across Formula 1 teams. Red Bull consistently excelled in achieving the fastest pit stops throughout this period, leveraging this as a key competitive advantage to secure crucial track positions and influence race outcomes. Ferrari demonstrated remarkable progress, transitioning from slower pit stops in earlier years (2012–2015) to becoming one of the more efficient teams by 2024, showcasing their focus on operational improvements.
#### Teams like Mercedes and McLaren maintained relatively consistent performance, though not at the level of Red Bull's efficiency. In contrast, mid-field and smaller teams such as Haas, Toro Rosso, and Williams exhibited variability, often struggling with slower pit stops due to resource constraints or operational inefficiencies. Additionally, the data indicates a general trend of faster pit stops across all teams over the years, reflecting advancements in technology, training, and strategy.
#### By 2024, pit stops have become a critical focus for teams, with nearly all striving for sub-15-second times, emphasizing their importance in an increasingly competitive F1 landscape. This period highlights the growing role of pit stop efficiency in shaping race strategies and overall championship outcomes.
#### c) Visualisation: Heatmap of pit stops by teams and seasons.
## Pitstop Images over years
![Fastest Pitstop Heatmap](https://drive.google.com/uc?id=19rEosOwneekueyVi-Oo3y-y_KL0tulXG)

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### 3.7) Analysis of Points Scored by Teams and Drivers
#### a) Objective: Analyze the distribution of points among drivers and constructors over different seasons.
#### b) Conclusion: The analysis of the points distribution among drivers and constructors highlights notable disparities in performance. Among drivers, Lewis Hamilton stands out as the highest point scorer with an impressive 4713.5 points, showcasing his dominance in modern Formula 1. Sebastian Vettel and Max Verstappen follow with 3098 and 2744.5 points, respectively, reflecting their consistent performances. In contrast, Daniel Ricciardo has the lowest total among the top 10 drivers, with 1319 points, indicating that while successful, he hasn't matched the sustained dominance of others in the chart.
#### For constructors, Ferrari emerges as the team with the highest points, totaling a staggering 10,772.27, cementing their legacy as one of the most successful teams in F1 history. Mercedes and Red Bull follow with 7502.64 and 7472 points, respectively, highlighting their modern-era dominance. At the lower end, Tyrrell, with 711 points, and Benetton, with 861.5 points, represent teams that have had historic success but lack the long-term consistency of Ferrari, Mercedes, or Red Bull. This distribution underscores the evolution of F1, with a mix of legendary teams and drivers at the pinnacle of the sport.
#### c) Visualisation: Horizontal bar plots of points scored by drivers and teams.
![Driver Points Table](https://drive.google.com/uc?id=19LcigpvQhF0UyjEyqG0t9mNxdrfxVFfn)
![Driver Points Chart](https://drive.google.com/uc?id=1541fpcbwVWQqcfVcdQR_BT570QymKKKu)
![Team Points Table](https://drive.google.com/uc?id=193rBbbj3OoLOFP1r-b8gn1TNVbEcZbpZ)
![Team Points Chart](https://drive.google.com/uc?id=1al8uCMxofo3opy1Cd87K3QBasdf4AUmu)

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### 3.8) Conversion from Pole to Win
#### a) Objective: Determine which drivers were most effective at converting pole positions into race wins.
#### b) Conclusion: Drivers such as Ayrton Senna and Lewis Hamilton demonstrated a high conversion rate, showcasing their race craft and control under pressure.
#### c) Visualisation: A pole-to-win ratio bar graph for the drivers.
## Pole to Win Analysis
![Pole to Win Analysis](https://drive.google.com/uc?id=1zXcxU1MyrEs8J0kXeKoOOASsiqdS9QpA)

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### 3.9) Race Statistics & Fastest Lap Records
#### a) Objective: Identify circuits and drivers associated with the fastest laps.
#### b) Conclusions
#### 1) The Fastest Lap Record Analysis
##### The Fastest Lap Record by Circuit chart and corresponding table provide a detailed overview of the fastest lap times recorded at various Grand Prix circuits, showcasing both driver skill and track characteristics. Notable performances include Kimi Räikkönen at the Belgian Grand Prix with the slowest lap time of 1:45.108, reflecting the circuit's long layout and challenging corners. In contrast, Max Verstappen holds the fastest lap time of 1:28.139 at the Eifel Grand Prix, showcasing his exceptional speed and the circuit's relatively short layout. Other standout performances include Lewis Hamilton dominating at multiple circuits like Singapore, Russian, and Saudi Arabian Grand Prix, emphasizing his versatility across different tracks. The table highlights the diversity of drivers and circuits, showcasing legendary performances by Michael Schumacher, Sebastian Vettel, and newer stars like Charles Leclerc and Oscar Piastri. This analysis emphasizes the combination of driver talent and circuit challenges in achieving record lap times.
![Fastest Lap Bar](https://drive.google.com/uc?id=1M5GhSf5vdzA_CfHkX4IfgGEYrICugLyr)
![Fastest Lap Table](https://drive.google.com/uc?id=1M_vy81c2OMD___KcAfXUrKP7Q7Sba9B3)

#### 2) Race Statistics
##### Over the years, Formula 1 has seen a steady increase in the number of races, as shown by the "Number of Races per Year" graph, indicating the sport's growing global appeal and expanded calendar.The "Top 10 Circuits with Most Races" chart demonstrates the historic significance of venues like the Italian and British Grand Prix, which have hosted the highest number of races, showcasing their long-standing place in the sport. Furthermore, the "Upcoming Races and Their Scheduled Months" chart provides a clear view of the F1 calendar, with races like Abu Dhabi Grand Prix typically concluding the season. Together, these charts underline the sport's historical evolution, global reach, and meticulous planning, showcasing its progression as a premier motorsport event.
![Number of Races per Year](https://drive.google.com/uc?id=19Z6viefWjc3vYbwlf3E7ozQ4hLjUvkmf)
![Circuits with Most Races](https://drive.google.com/uc?id=1cFdL3nSQGlBEpEwDwuQD3JzeV8U2aeiK)
![Upcoming Races](https://drive.google.com/uc?id=1w-0imx0Sf489J2dsB12hbgswl7yjWp_V)

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### 3.10) Pit Stop Strategy Optimisation
#### a) Objective: Identify the most influential factors in optimizing pit stop strategy.
#### b) Conclusions
               precision    recall  f1-score   support

           0       0.70      0.62      0.66      1654
           1       0.66      0.73      0.69      1648

      accuracy                         0.68      3302
     macro avg       0.68      0.68    0.68      3302
    weighted avg     0.68      0.68    0.68      3302
##### The best model is the LightGBM algorithm, tuned with hyperparameter tuning to achieve a 68% accuracy in choosing the best pit stop strategy. For Class 0, this had a precision of 70%, which shows that the model was correct about 70% of the time for the finishes that were not in the top 5 while having a recall of 62%; this means 62% of actual non-top-5 outcomes have been identified by the model. These are balanced by the F1-score, standing at 66%. In the case of Class 1, Top 5 Finish, the precision was 66%, with the recall being higher at 73%, showing that the model was very strong in identifying top-5 finishes, with an F1-score of 69%.
##### The macro and weighted average F1-scores of 68% show that the performance for both classes is balanced, considering class imbalance, which was effectively handled using SMOTE. That will say how strong the model is to cope with an imbalanced dataset. One possible line of improvement can be trying to improve Recall Class 0 and Overall Accuracy by engineering new data, trying advanced feature selections, or ensemble techniques. The following workflow finally sums up class imbalance handling, hyperparameter tuning, and explainability all combined to find an optimal strategy to make a pit stop in Formula 1.

#### c) Visualisation: Using SHAP Summary plots and Feature Importance Plots
#### 1) Analysis of Figures
#### 1.1) Feature Importance Plot: The first plot shows the relative importance of features used in the LightGBM model for the task of predicting optimal pit stop strategies. The most influencing feature is pit_duration, meaning that the time spent in the pit is highly important to predict if a top-5 finish is likely. Other features such as lat, lng, alt, and year feature, however, have lower importance though they are also contributing to this prediction.
![Feature Importance](https://drive.google.com/uc?id=1Q7x3bwMkfvqhta2Jv7dS69Mm4ZG3jHPk)

#### 1.2) What is SHAP Summary plot & Explanation of our results
#### Q) What is SHAP Summary plot?
##### SHAP (SHapley Additive exPlanations) is a tool based on game theory that explains machine learning predictions by quantifying each feature’s contribution. It enhances model transparency, identifies critical features, builds trust in complex models, and provides both global and local explanations, aiding interpretability and decision-making, particularly in high-stakes applications like Formula 1 strategy optimization.
#### Explanation of our Results
#### The SHAP plot provides an explainability perspective on how each feature values impact the model's prediction. The color red represents high feature values, while blue represents low feature values. Also, similar to the feature importance plot, pit_duration has the biggest impact, and a higher value, as represented in red, normally reduces the likelihood of top-5 finishes. Features such as year, alt, and lat have a more nuanced impact on the predictions, at times positively depending on their value or vice versa. This detailed visualization helps understand the direction and magnitude of feature effects.
![SHAP Summary Plot](https://drive.google.com/uc?id=1Pxr77FXZWacjzYM4O3Ctr9NgkNyczavt)

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### 3.11) Prediction of Top 5 Finish
#### a) Objective: Predict whether a driver will finish in the top 5 based on qualifying results, driver/team form, and track characteristics.
#### b) Conclusions
Accuracy: 0.86
ROC AUC: 0.91
Classification Report:
              precision    recall  f1-score   support

           0       0.90      0.92      0.91       802
           1       0.74      0.69      0.71       264

      accuracy                           0.86      1066
     macro avg       0.82      0.81      0.81      1066
    weighted avg     0.86      0.86      0.86      1066
##### The predictive model yielded an accuracy of 81% and a very strong ROC AUC score of 0.86, hence establishing its efficiency in predicting whether a driver will finish in the top 5. Precisely, from the classification report for Class 0 (Not Top 5 Finish), it has a precision of 86% and a recall of 90%, indicating that the model is good at identifying the correct non-top-5 finishes. Class 1, Top 5 Finish: The model did a relatively weak job with Class 1, having precision of 64% and recall of 55%, hence showing further room for improvement regarding capturing top-5 finishes.
##### The feature importance plot identifies the position of qualification as the most deciding factor in Top 5 finishes, followed by the year of the race. The track characteristics like latitude, longitude, and altitude have a lesser influence on the outcome. In summary, the model shows very strong predictive power and actionable insights into some key factors driving race outcomes; however, it can be further improved by enhancing its utility in handling Class 1 predictions.

#### c) Visualisation: Using feature importance and ROC-AUC Curve
#### 1) Conclusion of feature importance
##### The feature importance plot highlights qualifying position as the most influential predictor for top-5 finishes, followed by average pit stop time. Geographic and track-specific variables like latitude, longitude, and altitude have minimal impact, emphasizing the importance of race-day performance metrics in the prediction.
![Feature Importance Plot for Top 5 Finish](https://drive.google.com/uc?id=16MOKAWyyz4wP1zkX1xkL5ppf_LNIeODh)

#### 2) ROC-AUC Curve
##### The ROC-AUC curve demonstrates the model's strong predictive performance, achieving an AUC score of 0.91. This indicates that the RandomForestClassifier effectively differentiates between drivers likely to finish in the top 5 and those who won't, with a high degree of accuracy across various decision thresholds.
![ROC-AUC Curve](https://drive.google.com/uc?id=1qmju-fQ9DikbIGH8KholgskeEIXH7uNC)

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### 4) Data
### 4.1) Description of Datasets
#### The project is based on the historic Formula 1 data, where races, drivers, teams, circuits, qualifying results, pit stops, and championships are recorded with a high level of detail. In this dataset, Formula 1 is deeply represented. This allows analyzing and predicting multiple features. Its key components include:
    1) Race Data: Information about race locations, dates, and circuits.
    2) Driver Data: driver's name, nationality, and name of the allocated team.
    3) Team Data: Information about constructors (teams) and their performance.
    4) Qualifying Data: Results from qualifying sessions, including positions. 
    5) Pit Stop Data: Timings and details of pit stops during races. 
    6) Lap Times and Fastest Laps: Insights into the performance of drivers and cars during races.
    7)  Championship Standings: Points earned by drivers and teams across seasons.
#### This information is very important to find out the trends, identify performance drivers, and build predictive models for race outcomes, championship predictions, and strategy optimizations.

### 4.2) Sources of Data
#### The dataset used in this project is sourced from Kaggle, a well-known platform for sharing and analyzing datasets. The data is publicly available and meticulously curated by contributors passionate about Formula 1.
#### You can access the dataset on Kaggle at the following link: https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020/code

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### 5) Methodologies
### 5.1) Analytical Approaches
#### The project employs a variety of analytical techniques to explore historical Formula 1 data:
##### a) Descriptive Analytics:  The analysis of trends in circuit lap times, driver nationalities, and FIA championship rivalries.
##### b) Comparative Analysis: Looking into key moments like the heated 2021 championship and pole-to-win efficiency.
##### c) Visual Analytics: Results are intuitively represented through Treemaps, bar charts, scatter plots, and heatmaps.

### 5.2) Predictive Modeling Techniques
#### Advanced predictive models were built using machine learning methods:
##### a) Binary Classification Models: The models for predicting top 5 finishes and optimal pit stop strategies.
##### b) Multiclass Classification Models: Used for race outcome predictions.
##### c) Gradient Boosting Models: To predict driver champion winners.
##### d) Feature Importance Analysis: Identifying the critical factors influencing the predictions, such as qualifying position and pit stop duration.

### 5.3) Tools & Frameworks Used
##### a) Programming Language: Python
##### b) Visualization Libraries: Matplotlib, Seaborn, Plotly
##### c) Machine Learning Libraries: Scikit-learn, XGBoost, LightGBM
##### d) Data Processing Libraries: Pandas, NumPy
##### e) Development Environment: Docker-hosted Jupyter Notebooks
##### f) Version Control: Git and GitHub for collaborative tracking.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### 6) Results
#### 6.1) Inights Driven from Analysis
##### a) These would correspond to Spa-Francorchamps for the longest and Monaco for the shortest average lap times.
##### b) Key drivers and constructors dominate the pole-to-win efficiency statistics.
##### c) Pit stop analysis gives insight into strategy significance, while some features like pit duration play a critical role in top 5 finishes.
##### d) Nationalities are varied among drivers, although European drivers have been more represented throughout history.

#### 6.2) Key Outcomes for Predictive Modeling
##### a) Top 5 Finish Prediction: Achieved an accuracy of 86% with an ROC AUC score of 0.91, showcasing reliable performance.
##### b) Pit Stop Strategy Optimization: Identified pit stop duration as a key determinant for race strategy success.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### 7) Code Script
#### 7.1) The project is structured into modular scripts for easier maintenance:
##### a) driver_championship_prediction.py: Predicting the driver champion based on early-season data.
##### b) top5_finish_prediction.py: Make top 5 finish prediction using qualifying and pit stop information.
##### c) pit_stop_strategy_optimization.py - Analyses and optimizes pit stop strategies.
##### d) Analysis scripts: exploratory analysis in various scripts for gaining insights into insights like lap times and points distribution.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### 8) Requirements
#### The required Python libraries are listed in the requirements.txt file.
    1) To install the dependencies, use:
       pip install -r requirements.txt

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### 9) Acknowledgements
#### 9.1) Credit to Data Sources and Tools Used
##### a) Data Sources:  Historic Formula 1 data were downloaded from the Kaggle portal, which provided very detailed information about circuits, drivers, races, and results.
##### b) Tools and Libraries: 
      1) Python stacks like Pandas, Matplotlib, Seaborn, Plotly, Scikit-learn, XGBoost, LightGBM for data treatment, visualization, and predictive modeling.
      2) Docker because it hosted the Jupyter Notebook, which allowed a portable and replicable environment throughout the development.
      3) Spyder IDE: This will help convert Jupyter Notebook code into structured Python (.py) files for much cleaner and modular scripts.
##### c) Platform: Git and GitHub for version control and collaboration.
##### d) Special thanks to the Formula 1 community for such curated datasets.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### 10) Conclusions
#### This project is an example of the power of data-driven insights and predictive modeling in the world of Formula 1-from lap time analysis to pit stop strategies and predicting race outcomes to championships. It uses historical data to show trends and make reliable forecasts.
#### 10.1) Key takeaways include:
##### a) Effective visualizations highlighting critical insights, such as pole-to-win efficiency and driver nationalities.
##### b) Predictive models which achieve high accuracy and ROC AUC scores while giving actionable insights into top 5 finishes and season-long championship predictions.
##### c) Pit stop strategy optimization, showing crucial factors in race success, such as pit duration.
##### d) The modularity and reproducibility of this code structure make such analysis and models easy to extend into future seasons or apply on different motorsport datasets. It enhances one's understanding of the dynamics at play in Formula 1 and thus lays the foundational work for exploratory analytics on motorsport.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
