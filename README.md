Reinforcement Learning Trading Agent Exploration
CS394R/ECE381V Reinforcement Learning Report

CNNpredrep.ipynb Trains and produces the directional predictions

./Configs CNNpred Model configs

./dataprocessesing CNNpred Variable preprocess

./pkl_results commulative returns for each model (A2C, ensembled, ect)

./Results CNNpred Model Params for each year trained

The following are the training and backtesting code for their respective model(Ensemble, Ensemble w/ CNNpred, Individual(A2C, DDPG, PPO), Individual w/ CNNpred)
./FinRLEnsemble_WPRED.ipynb 
./FinRLEnsemble.ipynb
./FinRLEnsembleA2C.ipynb
./FinRLEnsembleA2CWPRED.ipynb
./FinRLEnsembleDDPG.ipynb
./FinRLEnsembleDDPGWPRED.ipynb
./FinRLEnsemblePPO.ipynb
./FinRLEnsemblePPOPRED.ipynb


FinRL CODE USED FROM https://github.com/AI4Finance-Foundation/FinRL
SOME FUNCTIONS BORROW FROM: https://github.com/hoseinzadeehsan/CNNpred-Keras