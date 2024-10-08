# Test

This folder contains the files for three different datasets:

- `kar`, $\sim 40$ nodes;
- `words`, $\sim 100$ nodes;
- `vote`, $\sim 1000$ nodes;

For each dataset, we have used three different Detection Algorithms (our $f(\cdot)$ function):

- **Greedy Modularity**;
- **Modularity**;
- **Walk Trap**;

For each combination of dataset and Detection Algorithm, we have performed two different experiments:

- **Node Deception**: Hide a node from its initial community;
  - In this case we test the Agent on:
    - three different values of the *Similarity Coefficient* $\tau = \{0.3, 0.5, 0.8\}$;
    - three different values of the *Budget Multiplier* $\beta = \{1, 2, 3\}$ (budget is equal to $\beta \times \text{avg degree of G}$);
  - Each combination of $\tau$ and $\beta$ is tested for 100 iterations, and we obtain:
    - **Goal**, percentage of achieved goal;
    - **NMI**, Normalized Mutual Information, value between 0 and 1;
    - Average number of **steps** to hide the node;
    - **Time** to hide the node;
- **Community Deception**: Hide a community, i.e. a set of nodes, distributing the nodes in the community among the other communities;
  - In this case we test the Agent on:
    - three different values of the *Similarity Coefficient* $\tau = \{0.3, 0.5, 0.8\}$;
    - three different values of the *Budget Percentage* $\beta = \{1\%, 3\%, 5\%\}$ (budget is equal to $\beta \times \text{number of edges}$);
  - Each combination of $\tau$ and $\beta$ is tested for 100 iterations, and we obtain:
  - **Deception Score**, metric defined in the paper, value between 0 and 1;
  - **NMI**, Normalized Mutual Information, value between 0 and 1;
  - Average number of **steps** to hide the node;
  - **Time** to hide the node;


## Table

### KAR

#### Node Deception

##### Greedy Modularity

Success Rate:

| $\tau$  |$\beta$| DRL-Agent (ours)  | Random              | Degree          | Roam          |
|---------|-------|-------------------|---------------------|-----------------|---------------|
| 0.3     | 1     | 32.33% ± 5.29     | 23.67% ± 4.81       | 9.33% ± 3.29    | 8% ± 3.07     |
| 0.3     | 3     | 51.33% ± 5.66     | 28% ± 5.08          | 11% ± 3.54      | 9% ± 3.24     |
| 0.3     | 5     | 64.33% ± 5.42     | 20% ± 5.28          | 12.67% ± 3.76   | 8.67% ± 3.18  |
| 0.3     | 10    | 83% ± 4.25        | 50% ± 5.66          | 12.67% ± 3.76   | 10.67% ± 3.49 |
| 0.5     | 1     | 28.33% ± 5.1%    | 25.33% ± 4.92% | 15.33% ± 4.08% | 12.67% ± 3.76% |
| 0.5     | 3     | 55.67% ± 5.62%   | 37.67% ± 5.48% | 12.33% ± 3.72% | 9.33% ± 3.29% |
| 0.5     | 5     | 70.33% ± 5.17%   | 44% ± 5.62%    | 19% ± 4.44%   | 16% ± 4.15%  |
| 0.5     | 10    | 88.67% ± 3.59%   | 59% ± 5.57%    | 21.67% ± 4.66% | 12.33% ± 3.72% |
| 0.8 | 1 | 40.67% ± 5.56%   | 34.33% ± 5.37% | 26.67% ± 5.00% | 22.00% ± 4.69% |
| 0.8 | 3 | 70.00% ± 5.19%   | 51.67% ± 5.65% | 21.67% ± 4.66% | 24.33% ± 4.86% |
| 0.8 | 5 | 82.33% ± 4.32%   | 59.00% ± 5.57% | 21.33% ± 4.64% | 28.33% ± 5.10% |
| 0.8 | 10 | 95.00% ± 2.47%  | 70.00% ± 5.19% | 31.67% ± 5.26% | 30.33% ± 5.20% |


Normalized Mutual Information:

| $\tau$  |$\beta$| DRL-Agent (ours)  | Random          | Degree        | Roam          |
|---------|-------|-------------------|-----------------|---------------|---------------|
| 0.3     | 1     | 0.9045 ± 0.0996  | 0.9125 ± 0.0939 | 0.9456 ± 0.0929 | 0.9518 ± 0.0926 |
| 0.3     | 3     | 0.787 ± 0.1128   | 0.8194 ± 0.1144 | 0.9689 ± 0.0511 | 0.955 ± 0.0733 |
| 0.3     | 5     | 0.72 ± 0.1146    | 0.7649 ± 0.1122 | 0.9665 ± 0.0534 | 0.9491 ± 0.0806 |
| 0.3     | 10    | 0.6684 ± 0.1059  | 0.704 ± 0.1083  | 0.9525 ± 0.0807 | 0.9317 ± 0.0976 |
| 0.5     | 1     | 0.905 ± 0.1016   | 0.9136 ± 0.095   | 0.951 ± 0.0871 | 0.9586 ± 0.0866 |
| 0.5     | 3     | 0.7837 ± 0.1186  | 0.8143 ± 0.1157 | 0.9749 ± 0.0469 | 0.953 ± 0.0817 |
| 0.5     | 5     | 0.7211 ± 0.1082  | 0.7442 ± 0.1215 | 0.9648 ± 0.0538 | 0.9463 ± 0.082  |
| 0.5     | 10    | 0.6566 ± 0.0985  | 0.7089 ± 0.113  | 0.9536 ± 0.0819 | 0.9375 ± 0.0963 |
| 0.8 | 1 | 0.9053 ± 0.0998  | 0.9083 ± 0.0955 | 0.9448 ± 0.0948 | 0.9535 ± 0.0949 |
| 0.8 | 3 | 0.7796 ± 0.1185  | 0.8082 ± 0.1181 | 0.9649 ± 0.0557 | 0.9523 ± 0.0738 |
| 0.8 | 5 | 0.7143 ± 0.1077  | 0.7687 ± 0.1164 | 0.9652 ± 0.0536 | 0.9381 ± 0.0873 |
| 0.8 | 10 | 0.6790 ± 0.0971 | 0.7188 ± 0.1130 | 0.9484 ± 0.0838 | 0.9277 ± 0.1079 |


##### Louvain

Success Rate:

| τ   | β | DRL-Agent (ours) | Random | Degree | Roam |
|-----|---|------------------|--------|--------|------|
| 0.3 | 1 | 17.00% ± 4.25%   | 14.33% ± 3.97% | 2.00% ± 1.58% | 8.67% ± 3.18% |
| 0.3 | 3 | 31.67% ± 5.26%   | 21.00% ± 4.61% | 3.00% ± 1.93% | 5.00% ± 2.47% |
| 0.3 | 5 | 45.00% ± 5.63%   | 29.67% ± 5.17% | 2.00% ± 1.58% | 5.00% ± 2.47% |
| 0.3 | 10 | 73.33% ± 5.00%  | 51.00% ± 5.66% | 1.00% ± 1.13% | 5.33% ± 2.54% |
| 0.5 | 1 | 32.00% ± 5.28%   | 27.33% ± 5.04% | 18.33% ± 4.38% | 20.67% ± 4.58% |
| 0.5 | 3 | 45.00% ± 5.63%   | 33.33% ± 5.33% | 17.33% ± 4.28% | 17.33% ± 4.28% |
| 0.5 | 5 | 56.67% ± 5.61%   | 40.67% ± 5.56% | 16.33% ± 4.18% | 18.67% ± 4.41% |
| 0.5 | 10 | 87.67% ± 3.72%  | 61.33% ± 5.51% | 18.67% ± 4.41% | 17.33% ± 4.28% |
| 0.8 | 1 | 97.00% ± 1.93%  | 95.00% ± 2.47% | 94.00% ± 2.69% | 95.33% ± 2.39% |
| 0.8 | 3 | 97.67% ± 1.71%  | 96.00% ± 2.22% | 96.67% ± 2.03% | 95.33% ± 2.39% |
| 0.8 | 5 | 98.67% ± 1.30%  | 96.00% ± 2.22% | 96.33% ± 1.97% | 98.33% ± 1.45% |
| 0.8 | 10 | 100.00% ± 0.00% | 95.67% ± 2.30% | 96.67% ± 2.03% | 95.67% ± 2.30% |

Normalized Mutual Information:

| τ   | β | DRL-Agent (ours) | Random | Degree | Roam |
|-----|---|------------------|--------|--------|------|
| 0.3 | 1 | 0.7455 ± 0.0741  | 0.7582 ± 0.0698 | 0.7667 ± 0.0718 | 0.7708 ± 0.0664 |
| 0.3 | 3 | 0.7435 ± 0.0722  | 0.7408 ± 0.0790 | 0.7633 ± 0.0700 | 0.7653 ± 0.0683 |
| 0.3 | 5 | 0.7237 ± 0.0745  | 0.7475 ± 0.0802 | 0.7672 ± 0.0714 | 0.7620 ± 0.0696 |
| 0.3 | 10 | 0.6945 ± 0.0892 | 0.7239 ± 0.0914 | 0.7638 ± 0.0711 | 0.7643 ± 0.0687 |
| 0.5 | 1 | 0.7451 ± 0.0682  | 0.7543 ± 0.0676 | 0.7622 ± 0.0694 | 0.7682 ± 0.0664 |
| 0.5 | 3 | 0.7375 ± 0.0721  | 0.7439 ± 0.0791 | 0.7615 ± 0.0684 | 0.7613 ± 0.0717 |
| 0.5 | 5 | 0.7217 ± 0.0793  | 0.7369 ± 0.0787 | 0.7731 ± 0.0662 | 0.7575 ± 0.0654 |
| 0.5 | 10 | 0.6976 ± 0.0843 | 0.7291 ± 0.0880 | 0.7666 ± 0.0714 | 0.7679 ± 0.0679 |
| 0.8 | 1 | 0.7491 ± 0.0709  | 0.7515 ± 0.0740 | 0.7706 ± 0.0689 | 0.7707 ± 0.0679 |
| 0.8 | 3 | 0.7357 ± 0.0723  | 0.7411 ± 0.0770 | 0.7703 ± 0.0681 | 0.7609 ± 0.0690 |
| 0.8 | 5 | 0.7189 ± 0.0838  | 0.7440 ± 0.0801 | 0.7638 ± 0.0675 | 0.7607 ± 0.0624 |
| 0.8 | 10 | 0.6990 ± 0.0903 | 0.7195 ± 0.0908 | 0.7743 ± 0.0682 | 0.7618 ± 0.0671 |

##### WalkTrap

Success Rate:

| τ   | β | DRL-Agent (ours) | Random | Degree | Roam |
|-----|---|------------------|--------|--------|------|
| 0.3 | 1 | 11.67% ± 3.63%  | 8.33% ± 3.13% | 0.00% ± 0.00% | 4.33% ± 2.30% |
| 0.3 | 3 | 36.67% ± 5.45%  | 27.67% ± 5.06% | 0.00% ± 0.00% | 4.33% ± 2.30% |
| 0.3 | 5 | 60.67% ± 5.53%  | 40.33% ± 5.55% | 0.00% ± 0.00% | 11.67% ± 3.63% |
| 0.3 | 10 | 83.67% ± 4.18% | 72.33% ± 5.06% | 0.00% ± 0.00% | 12.67% ± 3.76% |
| 0.5 | 1 | 18.33% ± 4.38%  | 13.00% ± 3.81% | 16.67% ± 4.22% | 5.33% ± 2.54% |
| 0.5 | 3 | 45.00% ± 5.63%  | 37.33% ± 5.47% | 16.00% ± 4.15% | 36.00% ± 5.43% |
| 0.5 | 5 | 70.00% ± 5.19%  | 51.00% ± 5.66% | 14.67% ± 4.00% | 28.67% ± 5.12% |
| 0.5 | 10 | 92.67% ± 2.95% | 76.67% ± 4.79% | 0.00% ± 0.00% | 29.67% ± 5.17% |
| 0.8 | 1 | 42.00% ± 5.59%  | 41.33% ± 5.57% | 45.00% ± 5.63% | 15.00% ± 4.04% |
| 0.8 | 3 | 68.00% ± 5.28%  | 59.33% ± 5.56% | 45.00% ± 5.63% | 44.33% ± 5.62% |
| 0.8 | 5 | 89.00% ± 3.54%  | 76.33% ± 4.81% | 41.00% ± 5.57% | 33.67% ± 5.35% |
| 0.8 | 10 | 98.00% ± 1.58% | 91.00% ± 3.24% | 16.67% ± 4.22% | 34.00% ± 5.36% |



Normalized Mutual Information:

| τ   | β | DRL-Agent (ours) | Random | Degree | Roam |
|-----|---|------------------|--------|--------|------|
| 0.3 | 1 | 0.8240 ± 0.0887  | 0.8267 ± 0.0864 | 0.7485 ± 0.0669 | 0.8452 ± 0.0788 |
| 0.3 | 3 | 0.7651 ± 0.0992  | 0.7703 ± 0.0874 | 0.7479 ± 0.0652 | 0.8707 ± 0.1208 |
| 0.3 | 5 | 0.7340 ± 0.0992  | 0.7462 ± 0.1002 | 0.7413 ± 0.0628 | 0.8543 ± 0.1019 |
| 0.3 | 10 | 0.7277 ± 0.0898 | 0.7216 ± 0.0930 | 0.9572 ± 0.0914 | 0.8513 ± 0.1014 |
| 0.5 | 1 | 0.8291 ± 0.0917  | 0.8326 ± 0.0884 | 0.7489 ± 0.0689 | 0.8484 ± 0.0773 |
| 0.5 | 3 | 0.7630 ± 0.0929  | 0.7869 ± 0.1047 | 0.7538 ± 0.0655 | 0.8588 ± 0.1212 |
| 0.5 | 5 | 0.7343 ± 0.1006  | 0.7485 ± 0.1062 | 0.7480 ± 0.0651 | 0.8449 ± 0.0989 |
| 0.5 | 10 | 0.7120 ± 0.0891 | 0.7245 ± 0.0942 | 0.9556 ± 0.0927 | 0.8547 ± 0.1031 |
| 0.8 | 1 | 0.8239 ± 0.0914  | 0.8256 ± 0.0891 | 0.7454 ± 0.0660 | 0.8496 ± 0.0804 |
| 0.8 | 3 | 0.7661 ± 0.0975  | 0.7758 ± 0.0927 | 0.7417 ± 0.0604 | 0.8682 ± 0.1199 |
| 0.8 | 5 | 0.7439 ± 0.0979  | 0.7402 ± 0.0934 | 0.7443 ± 0.0644 | 0.8508 ± 0.1003 |
| 0.8 | 10 | 0.7185 ± 0.0858 | 0.7284 ± 0.0887 | 0.9603 ± 0.0887 | 0.8438 ± 0.0997 |


