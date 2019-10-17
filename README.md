# Linear-Regression<br>_with_interaction_diminishing_return_terms
Make linear regression model with interaction and diminishing-return terms. Could get p-value of each terms


<h2>Documentation</h2>


>linear_regression. <strong>LinearRegression</strong>(DataFrame, target=None, interaction=False,<br>
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
diminishing_return=False, r_columns=None)

<h4>Parameters:</h4>

```
DataFrame: (required)Data in pandas dataframe format

target: (required)Target column in dataframe

interaction: boolean, False by default
             If True, add interaction terms

diminishing_return: boolean, False by default
                    If True, change input variables into diminishing-return terms.  ( 1 - e^(-r*x) ) / r
                    Adding r term

r_columns: array-like, optional
           If nothing entered, all input variables will be changed into diminishing-return term
           Choose columns you want to make diminishing-return
```

<h2>Example</h2>

```
from pandas as pd
from sklearn.preprocessing import scale
from linear_regression import LinearRegression

data = pd.read_csv('project1.2.csv')
data[['rebate','ad.spent','sales']] = scale(data[['rebate','ad.spent','sales']])
model = LinearRegression(data, target='sales', interaction = True, diminishing_return = True, r_columns=['rebate','ad.spent'])
model.train()
model.summary()
model.plot()
```

<p><strong>Result</strong></p>

![Example result](https://github.com/texasroh/Linear-Regression_with_interaction_diminishing_return_terms/blob/master/image/project%20result.PNG)




<h2>Academic Project</h2>
<p>Explanatory Data Analysis.<br>
Explain each components' effect to the sales.</p>
<p>Look at <strong>project 1.ipynb</strong> for more detail</p>
<p>Dataset: project1.2.csv  (Artificially generated for Academic purpose)</p>

<h3>Input Variables</h3>
<ul>
  <li>Rebate: $ amount for each purchase.</li>
  <li>Advertisement: MIL$ expenditure in each week.</li>
  <li>X-mas: last 6 weeks every year. Consider as Christmas season sales. (1: on-season, 0: off-season)</li>
</ul>

<h3>Target</h3>
<ul>
  <li>Sales: BIL$ sales in each week.</li>
</ul>

<h3>Full Model</h3>
![Full_model](https://github.com/texasroh/Linear-Regression_with_interaction_diminishing_return_terms/blob/master/image/full_model.PNG)

<h3>Restricted Model (Final Model)</h3>
![Restricted_model](https://github.com/texasroh/Linear-Regression_with_interaction_diminishing_return_terms/blob/master/image/restricted_model.PNG)
