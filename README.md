## CS260 HW03: Gradient Descent

Name: Juliette Ortiz

### Part 2: Analytic linear regression

Report the weights found for `sea_ice_data.csv` and `regression_train.csv`.
How closely do they agree with the linear models supplied in HW02?

Sea Ice Data ->  w: [ 1.90503822e+02 -9.22444610e-02]
Regression Train -> w: [ 2.44640709 -2.81635359]

### Part 3: Stochastic gradient descent

Record the settings and results from the normalized USA Housing experiment.

- Analytic cost: 204.9404512201138
- Analytic weights: [2.09175885e-14 6.51280605e-01 4.65062749e-01 3.43692235e-01
 5.77068966e-03 4.27271975e-01]
- Alpha: 0.0001
- Epsilon: 1e-10
- Maximum epochs:10000
- Completed epochs:10
- SGD cost: = 204.94902362497965
- SGD weights:[-4.37076377e-04  6.50468758e-01  4.65363014e-01  3.44347245e-01
  6.87482409e-03  4.27727905e-01]

Describe how you selected the SGD settings and what the cost-versus-epoch plot
shows about convergence.

I played around with the setting until the cost was nearly identical. I thought that a smaller learning rate would aid in a more refined search. 

### Part 4: Interpretation

1. How different are the analytic and SGD models?

Analytic Solution: 
y_hat = 0.651(x1) + 0.465(x2) + 0.344(x3) + 0.006(x4) + 0.428(x5) + 2.09e-14
SGD:
y_hat = 0.650(x1) + 0.465(x2) + 0.344(x3) + 0.0069(x4) + 0.(x5) -0.0004

The models are very similiar and the coefficients are nearly identical accross the features. 

2. Which feature has the largest effect in each model? Which features have
   little effect?

The analytic solution has x1 or avg area income to have the largest effect on the price of a house. The feature with the smallest effect is x4 or avg area/number of bedrooms. This aligns with 

3. Do these results make sense in the context of housing prices?

These results are intuitive in the context of housing prices when you consider the nature of cities where wealth and high income households are concentrated. It is easy to think of cities like New York City and Boston where housing is incredibly expensive because of the high demand of people trying to move in for job opportunities, public schools, etc. Compare this to rural areas with more space, less demand, and a larger working class population. 

4. Why does normalization matter when comparing weights and running SGD?

Normalization matters when comparing weights to have an accurate understanding of which features should have more impact on final predictions. SGD is sensitive to normalisation 

### Homework questionnaire

1. Approximately how many hours did this homework take?
10
2. How difficult was it from 1 (easy) to 5 (very difficult)?
3
3. What was the biggest challenge you faced?
I think making sure all the matrices and vectors were set up correctly was the hardest.
It was a lot of printing out shapes to make sure everything was compatible. 
# DSHW3
