import pandas as pd


df = pd.read_csv('./survey_results_public.csv')
schema = pd.read_csv('./survey_results_schema.csv')


num_vars = df[['Salary', 'CareerSatisfaction', 'HoursPerWeek', 'JobSatisfaction', 'StackOverflowSatisfaction']]

# Question 1 Part 1
# Drop the rows with missing salaries
drop_sal_df = num_vars.dropna(subset=['Salary'], axis=0)

# Question 1 Part 2
# Mean function
fill_mean = lambda col: col.fillna(col.mean())

# Fill the mean
fill_df = drop_sal_df.apply(fill_mean, axis=0)

# Question 2
rsquared_score = 0.03257139063404435
length_y_test = 1503
