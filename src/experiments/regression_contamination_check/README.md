## What
Another experiment regarding the possibility of LLMs to be contaminated, intentionally or not, with synthethic data. 
The idea is to record the performance of the model when knowing the dataset it is being tested on. If the performance increases significantly, then maybe it has seen that data before (together with the name)

## Why
The strong performance of LLMs on many datasets (e.g., `Friedman #1`) raises questions about potential prior exposure to these datasets during their training.
This is important, as if the LLMs have already seen the data, then the strong performance is not so surprising anymore.
In this experiment we attempt to provide additional experiments to answer this. 

Please note that we already included **5** new, original datasets, which are very unlikely to have been seen by the LLMs before. And the LLMs obtained strong performance on them. This experiment is just an addition to our contamination experiments.

## How
Record the model's performance when they know the dataset name. Then compare against the performance when they did not know the dataset name.