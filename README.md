# selcf_paper

This repository is forked from the original repository for the paper [Selection by Prediction with Conformal p-values](https://arxiv.org/abs/2210.01408).

## Folders 

- `utils/`: Python codes for the simulations. 
  - `gendata-model.py`: generate csv data for specified model (currently only 'rf' and 'mlp' supported).
  - `gendata-oracle.py`: generate csv data for the oracle model
  - `plot-model.py`: plot FDP, power, number of selection (`nsel`) and out-of-sample R^2 for specified model. The csv data need to be present.
  - `plot-oracle.py`: plot FDP, power, number of selection (`nsel`) and out-of-sample R^2 for the oracle model. The csv data need to be present.
  - `plot-trendcomparison.py`: Compare the trend for out-of-sample R^2 and power/number of selection.

## Sample arguments

python -u "d:\Github\selcf_paper\utils\gendata-model.py" -i 1000 -d 20 -n 100 mlp layers -r 1,21,1 hidden:8 

python -u "d:\Github\selcf_paper\utils\plot-model.py" -i 1000 -d 10 -n 100 rf max_depth -r 1,51,1 n_estim:50,max_features:sqrt 

python -u "d:\Github\selcf_paper\utils\plot-trendcomparison.py" -i 1000 -d 20 -n 100 mlp layers -r 1,11,1 hidden:8 

python -u "d:\Github\selcf_paper\utils\gendata-oracle.py" -i 1000 -d 20 -n 100