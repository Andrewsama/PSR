# PSR

This is an implemention for our Information Sciences paper based on Pytorch

[PSR Graph Neural Networks for Preference Social Recommendation]

by Gang-Feng Ma, Xu-Hua Yang, Yanbo Zhou, Xin-Li Xu


# Dataset
We use two datasets: [LastFM](https://grouplens.org/datasets/hetrec-2011/) and [Ciao](https://www.cse.msu.edu/~tangjili/datasetcode/truststudy.htm).

# Example to run the codes
1. Environment: I have tested this code with python3.8.5 Pytorch=1.7.1 CUDA=11.7 SCIPY=1.4.1 Sklearn=0.23.2  Pandas=1.1.3 numpy=1.19.5
2. Run PSR

    `python main.py --model=PSR --dataset=lastfm --decay=1e-4 --lr=0.001 --layer=3 --seed=2020 --topks="[10,20]" --recdim=64 --bpr_batch=2048`
