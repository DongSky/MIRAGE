# MIRAGE and Logos
MIRAGE: Assessing Hallucination in Multimodal Reasoning Chains of MLLM

[paper](https://arxiv.org/abs/2505.24238)
[data](https://huggingface.co/datasets/DongSky/mirage/tree/main)
[Logos_train_data](https://huggingface.co/datasets/DongSky/logos_train_data/tree/main)
[Logos-3B](https://huggingface.co/DongSky/Logos-3B/tree/main)
[Logos-7B](https://huggingface.co/DongSky/Logos-7B/tree/main)

## Eval
Our evaluation code is based on [VIC](https://github.com/Terry-Xu-666/visual_inference_chain)

1. Download eval data into eval code directory, then switch to this directory

2. execute following code for inference

```shell
python -m Vic.benchmark_test -p mirage.tsv -i original (for reasoning mllms)

or 

python -m Vic.benchmark_test -p mirage.tsv -i cot (for vanilla mllms)
```
3. Evaluation, specifically, for accuracy:
```shell
python -m Vic.benchmark_eval -b mirage -p output_inference_results.tsv
```

## Logos Train

We implement our train code based on [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF). We will upload our version soon. 

The training data has been released in [Logos_train_data](https://huggingface.co/datasets/DongSky/logos_train_data/tree/main).