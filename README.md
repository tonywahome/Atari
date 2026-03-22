# Atari DQN Hyperparameter Tuning — Results & Analysis

This project explores Deep Q-Network (DQN) hyperparameter tuning on Atari environments. Three team members — **Antony**, **Belyse**, and **Rodas** — each ran 10 experiments varying key hyperparameters to find the best-performing agent configuration.

**Best Models:**

- `antony/final_model.zip` — Experiment 9 (Mean Reward: **6.2**)
  - Antony's Model: [Google Drive Link](https://drive.google.com/file/d/1bbxf91jkFwwEmfpzOXTGll4CU38t_bDm/view?usp=sharing)
- `belyse/dqn_model.zip` — Experiment 2 (Mean Reward: **6.8**)
  - Belyse's Model: [Google Drive Link](https://drive.google.com/file/d/1lazEPfqNqHHzpi2zAghceipbFWfOY87G/view?usp=sharing)
- `rodas/bestmodel.zip` — Experiment 8 (Mean Reward: **7.8**)
  - Rodas's Model: [Google Drive Link](https://drive.google.com/file/d/1McdYxPAVvXNqmNxGLyLB989kOnpE-M-L/view?usp=sharing)

---

## Antony's Results

Best model: **`antony/final_model.zip`** (Experiment 9 — Balanced Aggressive)

|  Exp  | Policy        |    LR    | Gamma (γ) | Batch Size | Expl. Fraction | ε_start |  ε_end   | Mean Reward | Std Reward | Mean Ep Length | Time (min) | Description                        |
| :---: | ------------- | :------: | :-------: | :--------: | :------------: | :-----: | :------: | :---------: | :--------: | :------------: | :--------: | ---------------------------------- |
|   1   | CnnPolicy     |   1e-4   |   0.99    |     32     |      0.30      |   1.0   |   0.02   |    726.0    |   299.61   |      53.4      |    4.1     | Baseline with extended exploration |
|   2   | CnnPolicy     |   5e-4   |   0.99    |     32     |      0.30      |   1.0   |   0.02   |    558.0    |   111.79   |      65.0      |    4.1     | Higher learning rate               |
|   3   | CnnPolicy     |   1e-5   |   0.99    |     32     |      0.40      |   1.0   |   0.05   |    733.0    |   199.75   |     213.8      |    3.4     | Lower lr, longer exploration       |
|   4   | CnnPolicy     |   1e-4   |   0.95    |     32     |      0.30      |   1.0   |   0.02   |    276.0    |   129.63   |     161.0      |    4.7     | Lower gamma (myopic)               |
|   5   | CnnPolicy     |   1e-4   |   0.999   |     32     |      0.30      |   1.0   |   0.02   |    487.0    |   192.15   |      89.0      |    3.4     | Higher gamma (farsighted)          |
|   6   | CnnPolicy     |   1e-4   |   0.99    |     64     |      0.25      |   1.0   |   0.02   |    653.0    |   147.92   |     133.8      |    3.7     | Larger batch size                  |
|   7   | CnnPolicy     |   1e-4   |   0.99    |    128     |      0.25      |   1.0   |   0.02   |    570.0    |   164.98   |     137.8      |    6.3     | Even larger batch size             |
|   8   | CnnPolicy     |   1e-4   |   0.99    |     32     |      0.50      |   1.0   |   0.05   |    240.0    |   161.25   |      67.4      |    2.7     | Maximum exploration                |
| **9** | **CnnPolicy** | **2e-4** | **0.98**  |   **64**   |    **0.35**    | **1.0** | **0.03** |  **955.0**  | **214.58** |   **105.4**    |  **3.8**   | **Balanced aggressive**            |
|  10   | MlpPolicy     |   1e-4   |   0.99    |     32     |      0.30      |   1.0   |   0.02   |    458.0    |   169.34   |     457.8      |    0.4     | MLP on RAM observations            |

### Antony — Key Findings

- **Learning Rate:** Too high (5e-4, Exp 2) caused unstable learning and lower reward (558). Too low (1e-5, Exp 3) learned slowly but achieved comparable reward (733) to baseline. The sweet spot was 2e-4 (Exp 9).
- **Gamma:** Reducing gamma to 0.95 (Exp 4) made the agent myopic, resulting in the worst CNN performance (276). Increasing to 0.999 (Exp 5) was also detrimental (487). A slight reduction to 0.98 (Exp 9) struck the right balance.
- **Batch Size:** Increasing from 32 to 64 (Exp 6) or 128 (Exp 7) decreased performance, though 64 worked well when combined with other tuned parameters (Exp 9).
- **Exploration:** Excessive exploration (ε decay fraction=0.50, Exp 8) severely hurt performance (240). Moderate exploration (0.35) was optimal.

---

## Belyse's Results

Best model: **`belyse/dqn_model.zip`** (Experiment 2 — Very Low LR + Strong Exploration)

|  Exp  | Policy        |    LR    | Gamma (γ) | Batch Size | Expl. Fraction | ε_start |  ε_end   | Mean Reward | Std Reward | Mean Ep Length | Time (min) | Description                         |
| :---: | ------------- | :------: | :-------: | :--------: | :------------: | :-----: | :------: | :---------: | :--------: | :------------: | :--------: | ----------------------------------- |
|   1   | CnnPolicy     |   3e-4   |   0.97    |     48     |      0.35      |   1.0   |   0.01   |   1143.5    |   455.59   |     171.15     |   141.3    | High lr, conservative exploration   |
| **2** | **CnnPolicy** | **5e-5** | **0.99**  |   **32**   |    **0.45**    | **1.0** | **0.04** | **1427.0**  | **495.35** |   **319.55**   | **115.4**  | **Very low lr, strong exploration** |
|   3   | CnnPolicy     |   1e-4   |   0.90    |     32     |      0.30      |   1.0   |   0.02   |   1105.5    |   625.20   |     178.15     |   125.6    | Low gamma                           |
|   4   | CnnPolicy     |   2e-4   |   0.999   |     32     |      0.25      |   1.0   |   0.01   |   1134.0    |   421.07   |     248.65     |   113.4    | High gamma, fast learning           |
|   5   | CnnPolicy     |   1e-4   |   0.99    |     16     |      0.40      |   1.0   |   0.03   |    967.5    |   383.46   |     338.85     |    94.1    | Small batch size                    |
|   6   | CnnPolicy     |   1e-4   |   0.98    |     96     |      0.20      |   1.0   |   0.02   |   1142.0    |   423.37   |     228.65     |   188.1    | Large batch (stability focused)     |
|   7   | CnnPolicy     |   4e-4   |   0.96    |     64     |      0.45      |   1.0   |   0.05   |    943.5    |   293.64   |     170.15     |   131.0    | Aggressive learning + exploration   |
|   8   | CnnPolicy     |   8e-5   |   0.995   |     32     |      0.25      |   1.0   |  0.015   |   1175.0    |   274.89   |     311.95     |   106.1    | Conservative tuning                 |
|   9   | MlpPolicy     |   1e-4   |   0.99    |     32     |      0.35      |   1.0   |   0.02   |    558.5    |   153.08   |     1000.0     |    14.9    | MLP baseline                        |
|  10   | MlpPolicy     |   2e-4   |   0.97    |     64     |      0.40      |   1.0   |   0.03   |    799.5    |   242.52   |     580.8      |    13.9    | MLP with higher lr                  |

### Belyse — Key Findings

- **Learning Rate:** A very low lr of 5e-5 (Exp 2) produced the best result (1427), suggesting slow, stable gradient updates allow deeper learning. Higher lr (3e-4 in Exp 1, 4e-4 in Exp 7) still performed well but with more variance.
- **Gamma:** Lowering gamma to 0.90 (Exp 3) increased variance significantly (std=625.2). The standard 0.99 (Exp 2) performed best, while 0.999 (Exp 4) was slightly worse.
- **Batch Size:** The default batch size of 32 outperformed both smaller (16, Exp 5) and larger (96, Exp 6) sizes. Large batches added training time without proportional reward improvement.
- **Exploration:** Extended exploration (fraction=0.45, Exp 2) combined with a low learning rate was the winning strategy, giving the agent time to discover better policies before exploiting them.

---

## Rodas's Results

Best model: **`rodas/bestmodel.zip`** (Experiment 8 — Balanced Gamma)

|  Exp  | Policy        |    LR    | Gamma (γ) | Batch Size | Expl. Fraction | ε_start |  ε_end   | Mean Reward | Std Reward | Mean Ep Length | Time (min) | Description                                        |
| :---: | ------------- | :------: | :-------: | :--------: | :------------: | :-----: | :------: | :---------: | :--------: | :------------: | :--------: | -------------------------------------------------- |
|   1   | CnnPolicy     |   1e-4   |   0.99    |     32     |      0.30      |   1.0   |   0.02   |   1438.0    |   696.47   |     115.3      |    0.6     | Baseline configuration                             |
|   2   | CnnPolicy     |   1e-4   |   0.99    |     32     |      0.30      |   1.0   |   0.02   |   1868.0    |   537.23   |     149.3      |    0.88    | Smaller replay buffer (50k)                        |
|   3   | CnnPolicy     |   1e-4   |   0.99    |     32     |      0.30      |   1.0   |   0.02   |   1736.0    |   483.06   |     166.3      |    0.86    | More frequent updates (train_freq=1, grad_steps=4) |
|   4   | CnnPolicy     |   1e-4   |   0.99    |     32     |      0.30      |   1.0   |   0.02   |   1566.0    |   519.33   |     146.1      |    0.75    | Less frequent updates (train_freq=8)               |
|   5   | CnnPolicy     |   1e-4   |   0.99    |     32     |      0.30      |   1.0   |   0.02   |   1704.0    |   810.69   |     120.5      |    0.65    | Gradient clipping for stability                    |
|   6   | CnnPolicy     |   1e-4   |   0.99    |     32     |      0.30      |   1.0   |   0.02   |   1780.0    |   854.21   |     185.3      |    0.73    | Larger neural network architecture                 |
|   7   | CnnPolicy     |   1e-4   |   0.50    |     32     |      0.30      |   1.0   |   0.02   |    745.0    |   331.61   |      26.2      |    0.2     | Short-term focus (γ=0.50)                          |
| **8** | **CnnPolicy** | **1e-4** | **0.97**  |   **32**   |    **0.30**    | **1.0** | **0.02** | **2177.0**  | **792.10** |   **133.8**    |  **0.74**  | **Balanced gamma (γ=0.97)**                        |
|   9   | CnnPolicy     |   1e-4   |   0.99    |     32     |      0.05      |   1.0   |  0.005   |   1954.0    |   822.83   |      96.2      |    0.71    | Reduced exploration, early exploitation            |
|  10   | CnnPolicy     |   1e-4   |   0.99    |     16     |      0.30      |   1.0   |   0.02   |   1447.0    |   395.63   |     127.8      |    0.63    | Smaller batch size (16)                            |

### Rodas — Key Findings

- **Gamma:** The single most impactful parameter. Reducing gamma to 0.50 (Exp 7) was catastrophic (745 reward, shortest episodes). A slight reduction to 0.97 (Exp 8) produced the **best result across all three members** (2177), outperforming the default 0.99.
- **Replay Buffer & Update Frequency:** A smaller buffer (50k, Exp 2) improved performance (1868 vs 1438 baseline). More frequent updates (Exp 3, train_freq=1) also helped (1736), while less frequent updates (Exp 4, train_freq=8) were slightly worse.
- **Exploration:** Aggressive early exploitation (fraction=0.05, ε_end=0.005, Exp 9) achieved strong results (1954), suggesting that once a good policy is found, exploiting it quickly is beneficial.
- **Batch Size:** Halving batch size to 16 (Exp 10) slightly degraded performance compared to the default 32.

---

## Cross-Member Comparison

| Metric          | Antony (Exp 9) | Belyse (Exp 2) | Rodas (Exp 8) |
| --------------- | :------------: | :------------: | :-----------: |
| **Mean Reward** |     955.0      |     1427.0     |  **2177.0**   |
| Std Reward      |     214.58     |     495.35     |    792.10     |
| Policy          |   CnnPolicy    |   CnnPolicy    |   CnnPolicy   |
| Learning Rate   |      2e-4      |      5e-5      |     1e-4      |
| Gamma           |      0.98      |      0.99      |     0.97      |
| Batch Size      |       64       |       32       |      32       |
| Expl. Fraction  |      0.35      |      0.45      |     0.30      |
| ε_end           |      0.03      |      0.04      |     0.02      |

**Rodas's Experiment 8** achieved the highest mean reward (2177.0), though with higher variance. The key differentiator was **gamma = 0.97**, which slightly discounts distant future rewards — a practical trade-off that encourages the agent to prioritize achievable near-term rewards while still planning ahead.

---

## MLP vs CNN Comparison

All three members tested MLP (Multi-Layer Perceptron) against CNN (Convolutional Neural Network) policies:

| Member | CNN Best Reward | MLP Best Reward | CNN Policy | MLP Policy | Reward Drop |
| ------ | :-------------: | :-------------: | :--------: | :--------: | :---------: |
| Antony |  955.0 (Exp 9)  | 458.0 (Exp 10)  | CnnPolicy  | MlpPolicy  |  **-52%**   |
| Belyse | 1427.0 (Exp 2)  | 799.5 (Exp 10)  | CnnPolicy  | MlpPolicy  |  **-44%**   |
| Rodas  | 2177.0 (Exp 8)  |  — (CNN only)   | CnnPolicy  |     —      |      —      |

### Why CNN Outperforms MLP

- **Spatial feature extraction:** CNN processes raw pixel frames and can detect spatial patterns (enemy positions, projectile trajectories, ship location) through convolutional filters. MLP receives flattened RAM state or pixel vectors and lacks this spatial awareness.
- **Parameter efficiency:** CNN shares weights across the visual field via convolutional kernels, requiring far fewer parameters to capture the same spatial relationships that MLP must learn independently for each input position.
- **Training time trade-off:** MLP trains significantly faster (Antony: 0.4 min vs ~3-6 min; Belyse: ~14 min vs ~100-190 min) but achieves substantially lower rewards. The additional compute time for CNN training is justified by the performance gains.
- **Episode length:** MLP agents exhibited longer average episode lengths (Antony: 457.8; Belyse: 580.8–1000.0) despite lower scores, suggesting they survive longer but score less — indicating a passive/defensive policy rather than an optimal one.

**Conclusion:** CnnPolicy is the clear winner for Atari environments where visual observation is critical. MlpPolicy may serve as a fast baseline for initial experimentation but should not be used for final model training.

---

## Hyperparameter Insights Summary

| Hyperparameter             | Optimal Range | Key Takeaway                                                                                                              |
| -------------------------- | :-----------: | ------------------------------------------------------------------------------------------------------------------------- |
| **Learning Rate**          |  5e-5 – 2e-4  | Too high causes instability; too low slows convergence. Best results at moderate values.                                  |
| **Gamma (γ)**              |  0.97 – 0.99  | Slight reduction from 0.99 to 0.97 improved all results. Extreme values (0.50, 0.95) are harmful.                         |
| **Batch Size**             |      32       | Default 32 consistently performed well. Larger batches add compute cost without proportional gains.                       |
| **ε Exploration Fraction** |  0.25 – 0.45  | Moderate exploration is key. Too much (0.50) wastes training steps; too little (0.05) can work if other params are tuned. |
| **ε Final**                |  0.01 – 0.03  | Small residual exploration helps prevent getting stuck in local optima.                                                   |
