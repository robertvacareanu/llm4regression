## What
This folder contains the code for the experiment where we determine at what point do traditional supervised methods (e.g., Random Forest) start to perform better than LLMs on non-linear regression tasks.

## Why
We have shown empirically tha LLMs are capable of regression, sometimes even largely outshining traditional supervised methods. For example, on Original #1, out of the best 10 performing models, 6 are LLMs. And out of the top 5 best performing models, all are LLMs!
At least in part, this is surprising. LLMs, without any *direct* training can outperform supervised methods such as MLP, or Gradient Boosting. 
One natural hypothesis is: there is not enough data. However, the *same* amount of data is available to *all* methods.
Nevertheless, to further investigate whether there exists a point from where traditional supervised methods start to outperform 

## How
Similar to `regression_performance`, but instead of giving 50 (and repeating it 100 times), we give the models: 20 (input, output) pairs, then 50, 60, 70,0, 80, 90, 100, 150, 200, 250, 300, 400, 500 (and repeat 10 times each).