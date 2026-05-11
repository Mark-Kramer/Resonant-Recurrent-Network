# Resonant-Recurrent-Network

This repository contains code to perform the simulations and generate the figures in:

[Brain-inspired, interpretable, resonant recurrent neural networks](https://arxiv.org/abs/2506.17083)

---

The primary notebooks to estimate and apply the models are here:

| Notebook |  Run It |
| --- | --- |
| [Run the Resonant Recurrent Network (RRN)](./RRN_Results.ipynb) | Reproduce the **RRN** MNIST analysis (slow) and paper figures (fast) |
| [Run LSTM](./LSTM_Results.ipynb) | Reproduce the **LSTM** MNIST analysis (slow) |
| [Run Standard RNN](./Standard_RNN_Results.ipynb) | Reproduce the **Standard RNN** MNIST analysis (slow) |
| [Run spike train analysis](./RRN_Simulated.ipynb) | Reproduce spike train analysis all methods (slow) + paper figures (fast) |

These notebooks call helper functions (`.py` files) in this repository.
