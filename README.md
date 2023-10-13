This repository contains the code and data associated with **CoMPosT: Characterizing and Evaluating Caricature in LLM Simulations**, our EMNLP 2023 paper. If you have any questions, please contact me at: `myra [at] cs [dot] stanford [dot] edu`

# Code
- `get_caricature_scores.py`: script to run to compute individuation and exaggeration scores for a given dataset of simulations (example usage: `python get_caricature_scores.py examples/twitter_mini user comment`)
- `generation_scripts`: example scripts to generate simulations in different contexts 
- `topics`: lists of topics for each context.
- `generate_embeddings.ipynb`: compute embeddings for output data
- `individuation_scores.ipynb`: reproduce individuation score results
- `exaggeration_scores.ipynb`: reproduce exaggeration score results

# Data
`data`: generated simulations for the Online Forum and Interview contexts
