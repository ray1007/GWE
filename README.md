# Glyph-enhanced Word Embedding (GWE)

This is the code of the following [paper](https://arxiv.org/abs/1708.04755)
```
bib to be put... 
```    

This structure of this repository is described below: 
- [`convAE/`](./convAE) contains convolutional autoencoder which serves as the glyph feature extractor.
- ` ` is the extracted glyph features. (in txt format)
- [`data/`](./data) contains Traditional Chinese evaluation datasets. `240_trad.txt`, `297_trad.txt`, and `analogy.txt` were translated from [CWE](https://github.com/Leonard-Xu/CWE).
- RNN-based word embedding models, built with `tensorflow`.
- GWE code (written in C/C++). It extends [CWE](https://github.com/Leonard-Xu/CWE) with [MGE](http://www.aclweb.org/anthology/D/D16/D16-1100.pdf) and our proposed GWE.

Commands:
```python
make # compile GWE
```
