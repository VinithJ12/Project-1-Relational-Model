# FEMA Data Can Predict Which Disaster Survivors Need Help Most — Before an Inspector Ever Arrives

## When minutes matter, data can lead the way

Every year, hundreds of thousands of American families register for federal disaster housing assistance after hurricanes, floods, and storms tear through their communities. They fill out forms. They wait. And then they wait some more — for an inspector to come, for an eligibility decision to be made, for help to arrive. In the worst disasters, that wait stretches from days into weeks. This project asks a simple but powerful question: what if we didn't have to wait?

## The problem: a system built to react, not to predict

When a major disaster strikes, FEMA opens registration for Individual Assistance — a program that has served 5.6 million people and distributed over $4 billion in aid in a single year. But the process is fundamentally reactive. An inspector must physically visit each home before FEMA can determine whether it needs habitability repairs — the critical determination of whether a family can safely return. With inspectors stretched across thousands of square miles of disaster zones, high-need households can fall through the cracks. Research has shown that the poorest renters are 23% less likely to receive housing help than higher-income renters, and the poorest homeowners receive roughly half as much rebuilding assistance. The system doesn't just move slowly — it moves inequitably.

## The solution: let the data decide who goes first

Using FEMA's own historical records — 6.3 million household registrations from major disasters across Florida, Texas, Puerto Rico, and Louisiana — this project trained a machine learning model that can predict whether a household will require habitability repairs using only the information available at the moment of registration: household size, income, insurance status, residence type, and early damage indicators. The model achieves 81.2% accuracy, correctly identifying 4 out of 5 households that will need repairs — before an inspector ever arrives. A second model predicts the dollar value of property damage with an R² of 0.664, explaining nearly two-thirds of the variation in repair costs. Together, these tools could allow emergency managers to prioritize inspection schedules, pre-position repair crews, and flag the highest-need households for immediate follow-up.

## What the data revealed

The analysis uncovered stark geographic and demographic disparities in disaster damage outcomes. Louisiana had the highest habitability repair rate at 60.4% — meaning 6 in 10 inspected households in Louisiana disasters could not safely return home. Florida, by contrast, had a repair rate of just 19.9%. Homeowners experienced nearly double the repair rate of renters (47.6% vs 26.8%), despite earning on average $20,000 more per year — a finding that reflects the structural vulnerability of detached housing to severe weather. Flood damage was the single strongest predictor of habitability repairs: households with flood damage required repairs at nearly 7 times the rate of those without. The chart below shows the top factors the model identified as predictive of repair needs.

## Chart: what predicts whether a home needs repairs?

![Feature Importance](figures/feature_importance.png)

The feature importance chart from the Random Forest model indicates that personal property loss (`ppfvl`) is the most significant predictor of habitability repair requirements, followed by `geographic location` (state_encoded), `flood damage`, and `water level`.

Since these indicators are typically self-reported or estimated during the initial registration process, the model can generate habitability predictions the moment a household submits their FEMA application. This provides critical insight days before a physical inspection even takes place.
