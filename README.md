# NUT-RC: Noisy User-generated Text-oriented Reading Comprehension

## Introduction
**NUT-RC** is a noisy user-generated text-oriented reading comprehension model. First, it employs a set of text normalizers to transform the noisy and informal texts to the formal ones. Then, NUT-RC integrates the extractive and the generative RC model by a multi-task learning mechanism and an answer selection module. Overall, NUT-RC achieves state-of-the-art (SOTA) results on [TweetQA](https://tweetqa.github.io/) dataset.

For a detailed description of technical details and experimental results, please refer to our paper:

[NUT-RC: Noisy User-generated Text-oriented Reading Comprehension](url)<br/>
Rongtao Huang, Bowei Zou, Yu Hong, Wei Zhang, Ai Ti Aw, Guodong Zhou<br/>
COLING 2020

## Quickstart
### Setup Environment
1. python3.7.4
   
   Reference to download and install: [https://www.python.org/downloads/release/python-374/](https://www.python.org/downloads/release/python-374/)

2. install requirements (**Unfinish**)
   
   xx

### Train NUT-RC
1. Download data
    - The original TweetQA data is available at [TweetQA](https://tweetqa.github.io/) webset.
    - The normalized data can be found in [here](https://pan.baidu.com/s/1YtiSOzkYC9OLVH6bx1fJmA) (code: xxvl).
        > `SPLT`、`EXPN`、`WDLK` and `MISC` represents different text normalization described in NUT-RC paper.<br/>
        > For more information about text normalization, please refer to [normalise](https://github.com/EFord36/normalise).
 
2. Train a extractive RC model(Ext-RC)
    ```
    sh run_train_extractive_RC.sh
    ```
3. Train a generative RC model(Gen-RC)
    ```
    sh run_train_generative_RC.sh
    ```
4. Train a answer selection model
    ```
    sh run_train_answer_selection.sh
    ```
> The trained models reported in NUT-RC paper can be found in [here](https://pan.baidu.com/s/1gNiLwTuvbGA-aTtGvMwpjw) (code: 7bpl). <br/>
> **Note** that we ran experiments on 1 1080Ti GPU for all modules in NUT-RC.

### Evaluate NUT-RC
1. Prediction (**Unfinish**)
    ```
    sh prediction.sh
    ```
2. Evaluation
    ```
    python3 ./script/tweetqa_eval.py path_to_groundtruth_file path_to_prediction_file
    ```

## Notes and Acknowledgments
1. normlise is from: https://github.com/EFord36/normalise
2. UniLM is from: https://github.com/microsoft/unilm
3. Transformers is from: https://github.com/huggingface/transformers

## How do I cite MT-DNN?
```

```
