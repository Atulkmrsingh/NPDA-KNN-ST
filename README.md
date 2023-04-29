# NPDA-kNN-ST
Team members:
Atul Kumar Singh(22m0823)
Vishal Tapase(22m807)
Yashwant singh parihar(22m0798)
Sourabh kumar(22m0802)
Implementation of EMNLP'2022 paper "Non-Parametric Domain Adaptation for End-to-end Speech Translation".

Paper link: https://arxiv.org/pdf/2205.11211.pdf

Changes: Replacing kNN with rKNN with radius 0.5 on en-es dataset. Because RKNN uses a reduced set of training data to speed up the classification process. RKNN is useful in this case because data is high-dimensional and dataset size is very large and there are concerns about the memory requirements of the KNN algorithm. We have also used knn w.r.t cosine-similarity.

Training:
We trained using EuropalST dataset on the Fairseq S2T MuST-C predefined model and saved the key and value pair.

Inference:
We were facing the environment issue in the inference part. Omegaconf is throwing error due to its mismatch with the python version 3.11. We had changed the inference code in knnDatastore file but facing the issue in the environment variable.


# Flow

## Requirements

* python = 3.6
* pytorch = 1.8.1
* torchaudio = 0.8.1
* SoundFile = 0.10.3.post1
* numpy = 1.19.5
* omegaconf = 2.0.6
* PyYAML = 5.4.1
* sentencepiece = 0.1.96
* sacrebleu = 1.5.1
* faiss-gpu = 1.7.1.post1
* torch-scatter = 2.0.8

## Preparation of Data

Download Europarl-ST and Europarl-MT data for English-Spanish.
Run following script for preparing data:
``` bash 
bash myscripts/prep_europarst_data.sh
bash myscripts/prep_europarmt_data.sh
```

### Pre-trained Model and Data
Used the vocab file and pre-trained ST model provided by [Fairseq S2T MuST-C Example](https://github.com/pytorch/fairseq/blob/main/examples/speech_to_text/docs/mustc_example.md). 

## Inference with kNN Retrieval
### Create Datastore

Load the model for creating a cached datastore with the script as follow:

```bash
bash myscripts/mustc2europarlst/build_datastore.sh
```

### Build Faiss Index

The FAISS index requires a training stage where it learns a set of clusters for the keys. Once this is completed, the keys must all be added to the index. 

```bash
bash myscripts/mustc2europarlst/train_datastore.sh
```

### Inference via kNN-ST

```bash
bash myscripts/mustc2europarlst/eval_via_knn.sh
```

# Reference

We have implemented following paper "Non-Parametric Domain Adaptation for End-to-End Speech Translation".

```
@article{Du2022NonParametricDA,
  title={Non-Parametric Domain Adaptation for End-to-End Speech Translation},
  author={Yichao Du and Weizhi Wang and Zhirui Zhang and Boxing Chen and Tong Xu and Jun Xie and Enhong Chen},
  journal={ArXiv},
  year={2022},
  volume={abs/2205.11211}
}
```

