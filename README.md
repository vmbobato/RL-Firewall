# Reinforcement Learning Based Firewall

## Custom Gymnasium Environment for Sequential Network Security Decision-Making

This repository contains a custom Gymnasium reinforcement learning environment designed to train and evaluate adaptive firewall agents.
The goal is to simulate real-time packet/flow filtering decisions using an RL agent that learns a policy for ALLOW or DENY actions based on traffic features and sequential network behavior.

---

## Overview

Traditional firewalls and supervised ML classifiers treat each packet independently, ignoring how earlier decisions affect future network state. This environment models packet filtering as a sequential decision-making problem:  
- Blocking malicious traffic reduces attacker persistence  
- Allowing malicious traffic increases future threat level  
- Overblocking benign traffic increases interface load and affects normal traffic  
- Each decision influences future states, not just the current one  

This aligns with modern reinforcement-learning-based adaptive firewall research

---

## Environment Input Data

Raw columns:  
- `Src_IP`  
- `Flow_Byts/s`  (raw bytes-per-second for interface load)  

Binned (Discrete) features:  
- `Flow_Duration_bin`  
- `Tot_Fwd_Pkts_bin`  
- `Tot_Bwd_Pkts_bin`  
- `Pkt_Len_Mean_bin`  
- `Flow_Bytes_s_bin`  
- `ACK_Flag_Cnt_bin`  
- `Init_Fwd_Win_Byts_bin`  
- `SYN_Flag_Cnt_bin`  
- `syn_ratio_bin`  

Labels:  
- `label` -> `0` = benign, `1` = malicious  

All the preprocessing can be seen in `notebooks/`. The only dataset used in the environment is `data/preprocessed/Binerized_features.csv`.  

---

## Reward Function

The reward function models security outcomes and false positive/negative costs.  

If the flow is malicious (`label=1`)

| Action | Reward |
|----------|--------------|
| ALLOW (0) | -20 (false negative) |
| DENY (1) | +5 (correct block) |

If the flow is benign (`label=0`)

| Action | Reward |
|----------|--------------|
| ALLOW (0) | +0.5 (correct allow) |
| DENY (1) | -2 (false positive) |

This reward structure penalizes failing to block attacks heavily while still discouraging excessive blocking of legitimate traffic.

---

## Run the Code

First create and activate the Conda environment:
conda create -n rlfirewall
conda activate rlfirewall

Then install the dependencies:
pip install -r requirements.txt


Run random agent baseline:
python src/baseline_scripts/random_agent.py

Run rule based agent baseline:
python src/baseline_scripts/rule_based_agent.py

Run Q-Learning RL agent:
python src/agents/Q_Learning.py

Run SARSA RL agent:
python src/agents/SARSA.py.py

Run the Deep Q-Learning RL agent:
python src/agents/DQN.py

---

## Contributors

Vinicius Bobato  
Maria Visinescu