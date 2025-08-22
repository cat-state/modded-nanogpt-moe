# MoE fork of NanoGPT speedrun

This repository is a fork of the [NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt) to demonstrate the use of MoE for fun and profit. The fork is taken at the [10.8 min](https://github.com/KellerJordan/modded-nanogpt/blob/master/records/110324_UntieEmbed/d6b50d71-f419-4d26-bb39-a60d55ae7a04.txt) record. In the original speedrun, the goal is to get to less than 3.28 cross-entropy loss on the FineWeb dataset.

---

## Running the dense baseline
This is a modernized but simple causal transformer architecture, using rotary position embeddings, padded and untied token embeddings, RMSNorm, ReLU^2, QK-norm, and the Muon optimizer. This achieves 3.276 validation loss when trained for 2.4B tokens on the FineWeb dataset.

```bash
git clone https://github.com/cat-state/modded-nanogpt-moe && cd modded-nanogpt-moe
git checkout blogpost-version
pip install -r requirements.txt
python data/cached_fineweb10B.py 24
./run_dense.sh
```

## Running the MoE
This will train a top-k 1, 4 expert MoE. Each expert has the same architecture as the dense model, so the overall MoE is larger than the dense baseline, but has the same number of activated parameters. This achieves a 3.218 validation loss - an improvement of 1.8%. It gets to <3.28 val loss in 4000 steps, compared to 4500 for the baseline - an improvement of 11%.

```bash
git clone https://github.com/cat-state/modded-nanogpt-moe && cd modded-nanogpt-moe
git checkout blogpost-version
pip install -r requirements.txt
python data/cached_fineweb10B.py 24
./run_moe.sh
```

## References
0. [Daria Soboleva and Aman Tiwari. Debugging Dead MoE Models: A Step-by-step Guide](https://www.cerebras.ai/blog/moe-guide-debug)
1. [Keller Jordan and Jeremy Bernstein and Brendan Rappazzo and @fernbear.bsky.social and Boza Vlado and You Jiacheng and Franz Cesista and Braden Koszarsky and @Grad62304977. modded-nanogpt: Speedrunning the NanoGPT baseline](https://github.com/KellerJordan/modded-nanogpt)
2. [Guilherme Penedo, Hynek Kydlíček, Loubna Ben allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro Von Werra, Thomas Wolf. "The fineweb datasets: Decanting the web for the finest text data at scale." arXiv preprint arXiv:2406.17557 (2024).](https://arxiv.org/abs/2406.17557)

## Citation

```
@misc{modded_nanogpt_2024,
  author       = {Keller Jordan and Jeremy Bernstein and Brendan Rappazzo and
                  @fernbear.bsky.social and Boza Vlado and You Jiacheng and
                  Franz Cesista and Braden Koszarsky and @Grad62304977},
  title        = {modded-nanogpt: Speedrunning the NanoGPT baseline},
  year         = {2024},
  url          = {https://github.com/KellerJordan/modded-nanogpt}
}
```

