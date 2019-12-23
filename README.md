#### This is development repository of `xrnet` package (https://github.com/USCbiostats/xrnet), __survival module__. 

The program is the prototype of regularized Cox's proportional hazard hierachical model to incorporate external information. $y=(t, \delta)$ is survival outcome, $X$ is the conventional design matrix, having dimension $n\times p$. $Z$ is external data matrix, with dimension $p\times q$. In genomics study setting, it could be gene annotation data, which is featrues of the features in $X$. Modelling the two levels of data matrix with hierachical setting;
$$ h(t, x_i)=h_0(t)exp(x_i^T \beta) $$
$$ \beta = Z\alpha + \epsilon $$
where $\beta$ is the coefficients for first level features, and $\alpha$ is the coefficients for second level external features.

Then, integrate two levels of models into one objective function:
$$\min_{\beta,\alpha} -\frac{1}{n} \sum_k(\sum_{j\in D_k}x_j^T\beta-d_k ln(\sum_{j\in R_k}e^{x_j^T\beta})) + \frac{\lambda_1}{2}\vert\vert\beta-Z\alpha\vert\vert_2^2 + \lambda_2\vert\vert\alpha\vert\vert_1$$
The first component of the objective function is negative log Breslow's likelihood modeling 1st level information. The second component, $L_2$ norm of 2nd level information is to fit linear regression of external features on the coefficients of 1st level features. The third component is LASSO regularization on external feature coefficients $\alpha$, to do external feature selection. 

The algorithm uses proximal Newton algorithm and cyclic coordinate descent to optimize.

The `Rcpp` adpated Cpp function (to import to `R`), `cdlcoxRcpp()` is to fit the mdoel.
