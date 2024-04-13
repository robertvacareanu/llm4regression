This folder contains other examples of promtps. For example, `friedman2_s1000.txt` contains the prompts for `Friedman #2` with seed `1000`. 
It can be directly copy-pasted.

Below are the expected outputs.
- `friedman1_s1000.txt` -> `11.69` ([GPT-4](https://chat.openai.com/share/177571ad-3845-46a1-952f-963647620bea) predicts `12.89` in chat; Note that all experiments are over API)
- `friedman1_s1001.txt` -> `20.77` 
- `friedman1_s1002.txt` -> `12.95`
 
- `friedman2_s1000.txt` -> `689.01` ([GPT-4](https://chat.openai.com/share/78298975-19d5-4731-b29b-7a60fae88bd3) predicts `726.89` in chat; Note that all experiments are over API)
- `friedman2_s1001.txt` -> `221.98`
- `friedman2_s1002.txt` -> `300.2`
 
- `friedman3_s1000.txt` -> `1.47` ([GPT-4](https://chat.openai.com/share/fbb64727-3bf0-45ec-93ed-af6d4dbf8654) predicts `1.49` in chat; Note that all experiments are over API)
- `friedman3_s1001.txt` -> `1.41`
- `friedman3_s1002.txt` -> `1.45`

- `original1_s1000.txt` -> `80.39` ([GPT-4](https://chat.openai.com/share/808da995-99e6-444a-94da-fc7cd5ad49ff) predicts `83.63` in chat; Note that all experiments are over API; I included in `data/prompts/api_outputs/original1.jsonl` the output when calling the API)

For a shorter example with only 2 (input, output) pairs, please refer to `prompt.txt`. 