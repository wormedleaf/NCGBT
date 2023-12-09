## GCBT

Implementation of 'Graph Collaborative Filtering based Bug Triaging'
Modified from: https://gitee.com/papercodefromasexd/GCBT.git

### File Contents

- `preprocess.py` includes text cleaning, tokenization and parts.
- `dataset.py` includes dataset reading and slicing methods.
- `birnn.py` is the implementation of bi-directional RNN for bug report representation.
- `GCBT.py` is the implementation of spatial-temporal graph convolution model.
- `GCBTtrain.py` includes example method calls of GCBT.
- `trainer.py` is the implementation of a trainer for GCBT training.

### How to use
1. Please download the processed bug description at: https://pan.baidu.com/s/176RQxXw5zEbqxAY5-TFpdA?pwd=fd7e or https://mega.nz/file/oxUF0LxI#D39OZb3wctd1AfMOcG_Pt1KFAn4IzfQfttrJW6dqBNc and unzip it to the upper root directory.
2. Run `GCBTtrain.py`.
You can replace `eclipse` to `mozilla`, `office` in line 108 to change the dataset.

### Contribution

Any contribution (pull request etc.) is welcome.

