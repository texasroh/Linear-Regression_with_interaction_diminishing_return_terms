# Linear-Regression_with_interaction_diminishing_return_terms
Make linear regression model with interaction and diminishing-return terms. Could get p-value of each terms




<h2>Academic Project</h2>
<p>Explanatory Data Analysis.<br>
Explain each components' effect to the sales.</p>

Input Variables
<ul>
  <li>Rebate: $ amount for each purchase.</li>
  <li>Advertisement: MIL$ expenditure in each week.</li>
  <li>X-mas: last 6 weeks every year. Consider as Christmas season sales. (1: on-season, 0: off-season)</li>
</ul>

Target
<ul>
  <li>Sales: BIL$ sales in each week.</li>
</ul>

Full Model
<div>$Sales = c[0] + c[1] \left [ \frac{1-e^{-x[0]\cdot c[7]}}{b} \right ] + c[2] \left [ \frac{1-e^{-x[1]\cdot c[8]}}{b} \right ] + c[3]\cdot x[2] + c[4]( x[0]\cdot x[2])+c[5]( x[1]\cdot x[2])+c[6](x[0]\cdot x[1])$</div>

Restricted Model (Final Model)
<div>$Sales = c[0] + c[1] \left [ \frac{1-e^{-x[0]\cdot c[6]}}{b} \right ] + c[2] \cdot x[1]+ c[3]\cdot x[2] + c[4]( x[0]\cdot x[2])+c[5]( x[1]\cdot x[2])$</div>
