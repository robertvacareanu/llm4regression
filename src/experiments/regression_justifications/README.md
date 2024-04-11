## What
Get the predictions when not adding teh instruction to only give the output. This way, some models start explaining their prediction.

## Why
To analyze how models justify the final output and whether the explanations make sense.

## How
We simply give the prompt without the initial instruction to the model. For some models (e.g., Claude 3 Opus), this is enough to make them justify their prediction.