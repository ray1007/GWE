# Glyph-enhanced Word Embedding (GWE)

This is the code of the following [paper](https://arxiv.org/abs/1708.04755)
```
@InProceedings{su-lee:2017:EMNLP2017,
  author    = {Su, Tzu-ray  and  Lee, Hung-yi},
  title     = {Learning Chinese Word Representations From Glyphs Of Characters},
  booktitle = {Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing},
  month     = {September},
  year      = {2017},
  address   = {Copenhagen, Denmark},
  publisher = {Association for Computational Linguistics},
  pages     = {264--273},
  abstract  = {In this paper, we propose new methods to learn Chinese word representations.
	Chinese characters are composed of graphical components, which carry rich
	semantics. It is common for a Chinese learner to comprehend the meaning of a
	word from these graphical components. As a result, we propose models that
	enhance word representations by character glyphs. The character glyph features
	are directly learned from the bitmaps of characters by convolutional
	auto-encoder(convAE), and the glyph features improve Chinese word
	representations which are already enhanced by character embeddings. Another
	contribution in this paper is that we created several evaluation datasets in
	traditional Chinese and made them public.},
  url       = {https://www.aclweb.org/anthology/D17-1025}
}
```    

This structure of this repository is described below: 
- [`convAE/`](./convAE) contains convolutional autoencoder which serves as the glyph feature extractor.
- `char_glyph_feat.txt` is the extracted glyph features. (in txt format)
- [`data/`](./data) contains Traditional Chinese evaluation datasets. `240_trad.txt`, `297_trad.txt`, and `analogy.txt` were translated from [CWE](https://github.com/Leonard-Xu/CWE).
- GWE code (written in C). It extends [CWE](https://github.com/Leonard-Xu/CWE) with [MGE](http://www.aclweb.org/anthology/D/D16/D16-1100.pdf) and our proposed GWE.

Commands:
```python
# compile GWE
make 

# train GWE
./gwe -train data.txt -output-word vec.txt -output-char chr.txt -use-glyph 2 char-glyph char_glyph_feat.txt --size 300 -window 5 -sample 1e-4 -negative 5 -hs 0 -cbow 1 -cwe-type 2 -iter 3
```
