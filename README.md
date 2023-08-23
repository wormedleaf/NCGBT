This is the official PyTorch implementation for the NCGBT:
> Neighborhood Contrastive Learning based Graph Neural Network for Bug Triaging.

## Overview

We propose Neighborhood Contrastive learning based Graph neural network Bug Triaging framework, addressing data sparsity in graph bug Triaging using contrastive learning.

<div  align="center"> 
<img src="asset/intro.png" style="width: 75%"/>
</div>

## Requirements

```
recbole==1.1.1
python==3.10
pytorch==2.0.0
faiss-gpu==1.7.4
cudatoolkit==11.8.0
transformers==4.28.1
```

## Quick Start

```bash
python main.py --dataset eclipse
```

You can replace `eclipse` to `mozilla`, `office` to change the dataset.


