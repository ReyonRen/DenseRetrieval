# Awesome Research for Dense Retrieval


> A curated list of awesome papers related to dense retrieval.



## Table of Contents

- [Survey paper](#survey-paper)
- [Dense Retrieval with Ad-hoc Architecture](#dense-retrieval-with-ad-hoc-architecture)
  - [Late Interaction](#late-interaction)
  - [Incorporating Sparse Representation](#incorporating-sparse-representation)
- [Design Specific Training Procedure](#design-specific-training-precedure)
  - [Design Pre-training Task](#design-pre-training-task)
    - [Fine-tuning on Original Task](#fine-tuning-on-original-task)
    - [Fine-tuning on Other Task](#fine-tuning-on-other-task)
  - [Training with Expertal Task](#training-with-external-task)
    - [End-to-End Training](#end-to-end-trining)
    - [Multi-task Training](#multi-task-learning)
- [Data Construction and Sampling](#data-augmentation-and-sampling-strategy)
  - [Data Augmentation]
    - [Incorporating External Information](#incorporating-ecternal-information)
    - [Utilizing Related Model for Denoising and Distillation](#utilizing-related-model-for-denoising-and-distillation)
  - [Sampling Strategy]
- [Dataset](#dataset)
- [Other Resources](#other-resources)
  - [Some Retrieval Toolkits](#some-retrieval-toolkits)
  - [Other Summaries](#other-summaries)


 
## Survey Paper
- [Pretrained Transformers for Text Ranking: BERT and Beyond.](https://arxiv.org/abs/2010.06467) *Jimmy Lin et.al.*
- [Semantic Models for the First-stage Retrieval: A Comprehensive Review.](https://arxiv.org/pdf/2103.04831.pdf) *Yinqiong Cai et.al.*


## Dense Retrieval with Ad-hoc Architecture
- [DC-BERT: Decoupling Question and Document for Efficient Contextual Encoding.](https://arxiv.org/pdf/2002.12591.pdf) *Yuyu Zhang, Ping Nie et.al.* SIGIR 2020 short. (**DC-BERT**)
- [Learning To Retrieve: How to Train a Dense Retrieval Model Effectively and Efficiently.](https://arxiv.org/abs/2010.10469) *Jingtao Zhan et.al.* Arxiv 2020.
- [Sparse, Dense, and Attentional Representations for Text Retrieval](https://arxiv.org/pdf/2005.00181.pdf) *Yi Luan et.al.* TACL 2020. [[code](https://github.com/google-research/language/tree/master/language/multivec)] (**ME-BERT**)
- [Answering Complex Open-Domain Questions with Multi-Hop Dense Retrieval.](https://arxiv.org/pdf/2009.12756.pdf) *Wenhan Xiong at.el.* ICLR 2021 [[code](https://github.com/facebookresearch/multihop_dense_retrieval)]
- [XOR QA: Cross-lingual Open-Retrieval Question Answering.](https://arxiv.org/abs/2010.11856) *Akari Asai et.al.* NAACL 2021. [[code](https://nlp.cs.washington.edu/xorqa)]
- [Autoregressive Entity Retrieval.](https://arxiv.org/abs/2010.00904) *Nicola De Cao et.al.* ICLR 2021. [[code](https://github.com/facebookresearch/GENRE)]
- [A Replication Study of Dense Passage Retriever.](https://arxiv.org/pdf/2104.05740.pdf) *Xueguang Ma et.al.* Arxiv 2021.
- [The Curse of Dense Low-Dimensional Information Retrieval for Large Index Sizes.](https://arxiv.org/abs/2012.14210) *Nils Reimers et.al.* ACL 2021.

### Late Interaction
- [Poly-encoders: Architectures and pre-training strategies for fast and accurate multi-sentence scoring.](https://arxiv.org/pdf/1905.01969.pdf) *Samuel Humeau,Kurt Shuster et.al.* ICLR 2020. [[code](https://github.com/facebookresearch/ParlAI/tree/master/projects/polyencoder)] (**Poly-encoders**)
- [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT.](https://arxiv.org/pdf/2004.12832.pdf) *Omar Khattab et.al.* SIGIR 2020. [[code](https://github.com/stanford-futuredata/ColBERT)] (**ColBERT**)
- [Modularized Transfomer-based Ranking Framework.](https://arxiv.org/pdf/2004.13313.pdf) *Luyu Gao et.al.* EMNLP 2020. [[code](https://github.com/luyug/MORES)] (**MORES**)
### Incorporating Sparse Representation



## Design Specific Training Procedure

### Design Pre-training Task(s)
#### Fine-tuning on Original Task
- [Pre-training tasks for embedding-based large scale retrieval.](https://arxiv.org/pdf/2002.03932.pdf) *Wei-Cheng Chang et.al.* ICLR 2020. (**ICT, BFS and WLP**)
- PAIR. *Ruiyang Ren, Shangwen Lv et.al.* ACL 2021.
- [Is Your Language Model Ready for Dense Representation Fine-tuning?](https://arxiv.org/pdf/2104.08253.pdf) *Luyu Gao* Arxiv 2021. [[code](https://github.com/luyug/Condenser)]
#### Fine-tuning on Other Task
- [Latent Retrieval for Weakly Supervised Open Domain Question Answering.](https://arxiv.org/pdf/1906.00300.pdf) *Kenton Lee et.al.* ACL 2019. [[code](https://github.com/google-research/language/blob/master/language/orqa/README.md)] (**ORQA, ICT**)
- [REALM: Retrieval-Augmented Language Model Pre-Training.](https://arxiv.org/pdf/2002.08909.pdf) *Kelvin Guu, Kenton Lee et.al.* ICML 2020. [[code](https://github.com/google-research/language/blob/master/language/realm/README.md)] (**REALM**)
- [End-to-End Training of Neural Retrievers for Open-Domain Question Answering.](https://arxiv.org/abs/2101.00408) *Devendra Singh Sachan et.al.* ACL 2021. [[code](https://github.com/NVIDIA/Megatron-LM)]

### Training with External Task
#### End-to-End Training
- [Latent Retrieval for Weakly Supervised Open Domain Question Answering.](https://arxiv.org/pdf/1906.00300.pdf) *Kenton Lee et.al.* ACL 2019. [[code](https://github.com/google-research/language/blob/master/language/orqa/README.md)] (**ORQA, ICT**)
- [REALM: Retrieval-Augmented Language Model Pre-Training.](https://arxiv.org/pdf/2002.08909.pdf) *Kelvin Guu, Kenton Lee et.al.* ICML 2020. [[code](https://github.com/google-research/language/blob/master/language/realm/README.md)] (**REALM**)
- [Simple and Efficient ways to Improve REALM.](https://arxiv.org/abs/2104.08710.pdf) *Vidhisha Balachandran et.al.* Arxiv 2021. (**REALM++**)
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.](https://arxiv.org/abs/2005.11401) *Patrick Lewis et.al.* NIPS 2020. [[code](https://github.com/huggingface/transformers/blob/master/examples/rag/)] (**RAG**)
- [End-to-End Training of Neural Retrievers for Open-Domain Question Answering.](https://arxiv.org/abs/2101.00408) *Devendra Singh Sachan et.al.* ACL 2021. [[code](https://github.com/NVIDIA/Megatron-LM)]
#### Multi-task Training
- [Efﬁcient Retrieval Optimized Multi-task Learning.](https://arxiv.org/abs/2104.10129) *Hengxin Fun et.al.* Arxiv 2021.


## Data Construction and Sampling
### Data Augmentation
#### Incorporating External Information
#### Utilizing Related Model for Denoising and Distillation
- [Unified Open-Domain Question Answering with Structured and Unstructured Knowledge.](https://arxiv.org/pdf/2012.14610.pdf) *Barlas Oguz et.al.* Arxiv 2020.
- [Generation-Augmented Retrieval for Open-domain Question Answering.](https://arxiv.org/abs/2009.08553) *Yuning Mao et.al.* ACL 2021.
- [Neural Retrieval for Question Answering with Cross-Attention Supervised Data Augmentation.](https://arxiv.org/abs/2009.13815) *Yinfei Yang et.al.* Arxiv 2020.
- [Dense Passage Retrieval for Open-Domain Question Answering.](https://arxiv.org/pdf/2004.04906.pdf) *Vladimir Karpukhin,Barlas Oguz et.al.* EMNLP 2020 [[code](https://github.com/facebookresearch/DPR)] (**DPR**)
- [RepBERT: Contextualized Text Embeddings for First-Stage Retrieval.](https://arxiv.org/pdf/2006.15498.pdf) *Jingtao Zhan et.al.* Arxiv 2020. [[code](https://github.com/jingtaozhan/RepBERT-Index)] (**RepBERT**)
- [Optimizing Dense Retrieval Model Training with Hard Negatives.](https://arxiv.org/abs/2104.08051) *Jingtao Zhan et.al.* SIGIR 2021. [[code](https://github.com/jingtaozhan/DRhard)]
- [Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval.](https://arxiv.org/pdf/2007.00808.pdf) *Lee Xiong, Chenyan Xiong et.al.* Arxiv 2020. [[code](https://github.com/microsoft/ANCE)] (**ANCE**)
- [Neural Passage Retrieval with Improved Negative Contrast.](https://arxiv.org/abs/2010.12523) *Jing Lu et.al.* Arxiv 2020. 
- [RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering.](https://arxiv.org/pdf/2010.08191.pdf) *Yingqi Qu et.al.* NAACL 2021. (**RocketQA**)
- [Augmented SBERT: Data Augmentation Method for Improving Bi-Encoders for Pairwise Sentence Scoring Tasks.](https://arxiv.org/abs/2010.08240) *Nandan Thakur et.al.* NAACL 2021.
- [Unsupervised Document Expansion for Information Retrieval with Stochastic Text Generation.](https://arxiv.org/abs/2105.00666) *Soyeong Jeong et.al.* NAACL2021.
- [Efficiently Teaching an Effective Dense Retriever with Balanced Topic Aware Sampling.](https://arxiv.org/pdf/2104.06967.pdf) *Sebastian Hofstätter et.al.* SIGIR 2021.[[code](https://github.com/sebastian-hofstaetter/tas-balanced-dense-retrieval)] (**TAS-Balanced**)
- PAIR. *Ruiyang Ren, Shangwen Lv et.al.* ACL 2021.
- [Is Retriever Merely an Approximator of Reader?](https://arxiv.org/pdf/2010.10999.pdf) *Sohee Yang et.al.* Arxiv 2020.
- [Distilling Knowledge from Reader to Retriever for Question Answering.](https://openreview.net/pdf?id=NTEz-6wysdb) *Gautier Izacard, Edouard Grave* ICLR 2021. [[code](github.com/facebookresearch/FiD)]

### Sampling Strategy

## Datasets for Dense Retrieval

## Other Resources


### Some Retrieval Toolkits
- [Faiss: a library for efficient similarity search and clustering of dense vectors](https://github.com/facebookresearch/faiss)
- [Pyserini: a Python Toolkit to Support Sparse and Dense Representations](https://github.com/castorini/pyserini/)
- [MatchZoo: a library consisting of many popular neural text matching models](https://github.com/NTMC-Community/MatchZoo)
- [Anserini: Enabling the Use of Lucene for Information Retrieval Research](https://github.com/castorini/anserini)

### Other Summaries
- [awesome-pretrained-models-for-information-retrieval](https://github.com/Albert-Ma/awesome-pretrained-models-for-information-retrieval)
