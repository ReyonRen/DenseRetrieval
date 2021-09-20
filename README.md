# Awesome Research for Dense Retrieval


> A curated list of awesome papers related to dense retrieval.



## Table of Contents

- [Survey paper](#survey-paper)
- [Training Procedure](#training-procedure)
  - [Design Pre-training Tasks](#design-pre-training-tasks)
  - [End-to-End Training](#end-to-end-training)
  - [Multi-task Training](#multi-task-training)
- [Data Augmentation](#data-augmentation)
  - [Sampling Strategies](#sampling-strategies)
  - [Utilizing External Information](#utilizing-external-information)
  - [Utilizing Related Model for Denoising and Distillation](#utilizing-related-model-for-denoising-and-distillation)
- [Dense Retrieval for Downstream Applications](#dense-retrieval-for-downstream-applications)
  - [Question Answering](#question-answering)
  - [Re-ranking](#re-ranking)
- [Tailored Architecture](#tailored-architecture)
  - [Late Interaction](#late-interaction)
  - [Incorporating Sparse Retrieval](#incorporating-sparse-retrieval)
- [Dataset](#dataset)
- [Other Resources](#other-resources)
  - [Some Retrieval Toolkits](#some-retrieval-toolkits)
  - [Other Summaries](#other-summaries)


 
## Survey Paper
- [Pretrained Transformers for Text Ranking: BERT and Beyond.](https://arxiv.org/abs/2010.06467) *Jimmy Lin et al.*
- [Semantic Models for the First-stage Retrieval: A Comprehensive Review.](https://arxiv.org/pdf/2103.04831.pdf) *Yinqiong Cai et al.*





## Training Procedure

### Design Pre-training Tasks
- [Pre-training tasks for embedding-based large scale retrieval.](https://arxiv.org/pdf/2002.03932.pdf) *Wei-Cheng Chang et al.* ICLR 2020. (**ICT, BFS and WLP**)
- [Is Your Language Model Ready for Dense Representation Fine-tuning?](https://arxiv.org/pdf/2104.08253.pdf) *Luyu Gao* EMNLP 2021. [[code](https://github.com/luyug/Condenser)]
- [Unsupervised Corpus Aware Language Model Pre-training for Dense Passage Retrieval](https://arxiv.org/pdf/2108.05540.pdf) *Luyu Gao* Arxiv 2021. [[code](https://github.com/luyug/Condenser)]
- [Latent Retrieval for Weakly Supervised Open Domain Question Answering.](https://arxiv.org/pdf/1906.00300.pdf) *Kenton Lee et al.* ACL 2019. [[code](https://github.com/google-research/language/blob/master/language/orqa/README.md)] (**ORQA, ICT**)
- [REALM: Retrieval-Augmented Language Model Pre-Training.](https://arxiv.org/pdf/2002.08909.pdf) *Kelvin Guu, Kenton Lee et al.* ICML 2020. [[code](https://github.com/google-research/language/blob/master/language/realm/README.md)] (**REALM**)
- [Simple and Efficient ways to Improve REALM.](https://arxiv.org/abs/2104.08710.pdf) *Vidhisha Balachandran et al.* Arxiv 2021. (**REALM++**)
- [End-to-End Training of Neural Retrievers for Open-Domain Question Answering.](https://arxiv.org/abs/2101.00408) *Devendra Singh Sachan et al.* ACL 2021. [[code](https://github.com/NVIDIA/Megatron-LM)]
- [PAIR: Leveraging Passage-Centric Similarity Relation for Improving Dense Passage Retrieval.](https://aclanthology.org/2021.findings-acl.191/) *Ruiyang Ren, Shangwen Lv et al.* ACL 2021. [[code](https://github.com/PaddlePaddle/Research/tree/master/NLP/ACL2021-PAIR)]
- [Domain-matched Pre-training Tasks for Dense Retrieval](https://arxiv.org/pdf/2107.13602.pdf) *Barlas Oguz, Kushal Lakhotia and Anchit Gupta et al.* Arxiv 2021.

### End-to-End Training
- [Latent Retrieval for Weakly Supervised Open Domain Question Answering.](https://arxiv.org/pdf/1906.00300.pdf) *Kenton Lee et al.* ACL 2019. [[code](https://github.com/google-research/language/blob/master/language/orqa/README.md)] (**ORQA, ICT**)
- [REALM: Retrieval-Augmented Language Model Pre-Training.](https://arxiv.org/pdf/2002.08909.pdf) *Kelvin Guu, Kenton Lee et al.* ICML 2020. [[code](https://github.com/google-research/language/blob/master/language/realm/README.md)] (**REALM**)
- [Simple and Efficient ways to Improve REALM.](https://arxiv.org/abs/2104.08710.pdf) *Vidhisha Balachandran et al.* Arxiv 2021. (**REALM++**)
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.](https://arxiv.org/abs/2005.11401) *Patrick Lewis et al.* NIPS 2020. [[code](https://github.com/huggingface/transformers/blob/master/examples/rag/)] (**RAG**)
- [End-to-End Training of Neural Retrievers for Open-Domain Question Answering.](https://arxiv.org/abs/2101.00408) *Devendra Singh Sachan et al.* ACL 2021. [[code](https://github.com/NVIDIA/Megatron-LM)]
- [Jointly Optimizing Query Encoder and Product Quantization to Improve Retrieval Performance](https://arxiv.org/pdf/2108.00644.pdf) *Jingtao Zhan et al.* CIKM 2021.
### Multi-task Training
- [Efﬁcient Retrieval Optimized Multi-task Learning.](https://arxiv.org/abs/2104.10129) *Hengxin Fun et al.* Arxiv 2021.



## Data Augmentation

### Sampling Strategies
- [Learning To Retrieve: How to Train a Dense Retrieval Model Effectively and Efficiently.](https://arxiv.org/abs/2010.10469) *Jingtao Zhan et al.* Arxiv 2020.
- [Dense Passage Retrieval for Open-Domain Question Answering.](https://arxiv.org/pdf/2004.04906.pdf) *Vladimir Karpukhin,Barlas Oguz et al.* EMNLP 2020 [[code](https://github.com/facebookresearch/DPR)] (**DPR**)
- [RepBERT: Contextualized Text Embeddings for First-Stage Retrieval.](https://arxiv.org/pdf/2006.15498.pdf) *Jingtao Zhan et al.* Arxiv 2020. [[code](https://github.com/jingtaozhan/RepBERT-Index)] (**RepBERT**)
- [Optimizing Dense Retrieval Model Training with Hard Negatives.](https://arxiv.org/abs/2104.08051) *Jingtao Zhan et al.* SIGIR 2021. [[code](https://github.com/jingtaozhan/DRhard)]
- [Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval.](https://arxiv.org/pdf/2007.00808.pdf) *Lee Xiong, Chenyan Xiong et al.* Arxiv 2020. [[code](https://github.com/microsoft/ANCE)] (**ANCE**)
- [Neural Passage Retrieval with Improved Negative Contrast.](https://arxiv.org/abs/2010.12523) *Jing Lu et al.* Arxiv 2020. 
- [RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering.](https://arxiv.org/pdf/2010.08191.pdf) *Yingqi Qu et al.* NAACL 2021. (**RocketQA**)
- [Efficiently Teaching an Effective Dense Retriever with Balanced Topic Aware Sampling.](https://arxiv.org/pdf/2104.06967.pdf) *Sebastian Hofstätter et al.* SIGIR 2021.[[code](https://github.com/sebastian-hofstaetter/tas-balanced-dense-retrieval)] (**TAS-Balanced**)
- [Learning Dense Representations of Phrases at Scale.](https://arxiv.org/pdf/2012.12624.pdf) *Jinhyuk Lee, Danqi Chen et al.* ArxiV 2021. [[code](https://github.com/jhyuklee/DensePhrases)] (**DensePhrases**)
- [More Robust Dense Retrieval with Contrastive Dual Learning.](https://arxiv.org/abs/2107.07773) *Yizhi Li et al.* ICTIR 2021. [[code](https://github.com/thunlp/DANCE)] (**DANCE**)


### Utilizing External Information
- [Generation-Augmented Retrieval for Open-domain Question Answering.](https://arxiv.org/abs/2009.08553) *Yuning Mao et al.* ACL 2021.
- [Unified Open-Domain Question Answering with Structured and Unstructured Knowledge.](https://arxiv.org/pdf/2012.14610.pdf) *Barlas Oguz et al.* Arxiv 2020.
- [Unsupervised Document Expansion for Information Retrieval with Stochastic Text Generation.](https://arxiv.org/abs/2105.00666) *Soyeong Jeong et al.* NAACL2021.
- [Neural Passage Retrieval with Improved Negative Contrast.](https://arxiv.org/abs/2010.12523) *Jing Lu et al.* Arxiv 2020. 
- [PAIR: Leveraging Passage-Centric Similarity Relation for Improving Dense Passage Retrieval.](https://aclanthology.org/2021.findings-acl.191/) *Ruiyang Ren, Shangwen Lv et al.* ACL 2021. [[code](https://github.com/PaddlePaddle/Research/tree/master/NLP/ACL2021-PAIR)]


### Utilizing Related Model for Denoising and Distillation
- [Neural Retrieval for Question Answering with Cross-Attention Supervised Data Augmentation.](https://arxiv.org/abs/2009.13815) *Yinfei Yang et al.* Arxiv 2020.
- [RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering.](https://arxiv.org/pdf/2010.08191.pdf) *Yingqi Qu et al.* NAACL 2021. (**RocketQA**)
- [Augmented SBERT: Data Augmentation Method for Improving Bi-Encoders for Pairwise Sentence Scoring Tasks.](https://arxiv.org/abs/2010.08240) *Nandan Thakur et al.* NAACL 2021.
- [PAIR: Leveraging Passage-Centric Similarity Relation for Improving Dense Passage Retrieval.](https://aclanthology.org/2021.findings-acl.191/) *Ruiyang Ren, Shangwen Lv et al.* ACL 2021. [[code](https://github.com/PaddlePaddle/Research/tree/master/NLP/ACL2021-PAIR)]
- [Is Retriever Merely an Approximator of Reader?](https://arxiv.org/pdf/2010.10999.pdf) *Sohee Yang et al.* Arxiv 2020.
- [Distilling Knowledge from Reader to Retriever for Question Answering.](https://openreview.net/pdf?id=NTEz-6wysdb) *Gautier Izacard, Edouard Grave* ICLR 2021. [[code](github.com/facebookresearch/FiD)]
- [Efficiently Teaching an Effective Dense Retriever with Balanced Topic Aware Sampling.](https://arxiv.org/pdf/2104.06967.pdf) *Sebastian Hofstätter et al.* SIGIR 2021.[[code](https://github.com/sebastian-hofstaetter/tas-balanced-dense-retrieval)] (**TAS-Balanced**)
- [Distilling Knowledge for Fast Retrieval-based Chat-bots.](https://arxiv.org/pdf/2004.11045.pdf) *Amir Vakili Tahami et al.* SIGIR 2020. [[code](https://github.com/KamyarGhajar/DistilledNeuralResponseRanker)] (**Distill from cross-encoders to bi-encoders**)
- [Improving Efficient Neural Ranking Models with Cross-Architecture Knowledge Distillation.](https://arxiv.org/pdf/2010.02666.pdf) *Sebastian Hofstätter et al.* Arxiv 2020. [[code](https://github.com/sebastian-hofstaetter/neural-ranking-kd)] (**Distill from BERT ensemble**)
- [Distilling Dense Representations for Ranking using Tightly-Coupled Teachers.](https://arxiv.org/pdf/2010.11386.pdf) *Sheng-Chieh Lin, Jheng-Hong Yang, Jimmy Lin.* Arxiv 2020. [[code](https://github.com/castorini/pyserini/blob/master/docs/experiments-tct_colbert.md)] (**TCTColBERT: distill from ColBERT**)




## Tailored Architecture
### Late Interaction
- [Poly-encoders: Architectures and pre-training strategies for fast and accurate multi-sentence scoring.](https://arxiv.org/pdf/1905.01969.pdf) *Samuel Humeau,Kurt Shuster et al.* ICLR 2020. [[code](https://github.com/facebookresearch/ParlAI/tree/master/projects/polyencoder)] (**Poly-encoders**)
- [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT.](https://arxiv.org/pdf/2004.12832.pdf) *Omar Khattab et al.* SIGIR 2020. [[code](https://github.com/stanford-futuredata/ColBERT)] (**ColBERT**)
- [Modularized Transfomer-based Ranking Framework.](https://arxiv.org/pdf/2004.13313.pdf) *Luyu Gao et al.* EMNLP 2020. [[code](https://github.com/luyug/MORES)] (**MORES**)
- [DC-BERT: Decoupling Question and Document for Efficient Contextual Encoding.](https://arxiv.org/pdf/2002.12591.pdf) *Yuyu Zhang, Ping Nie et al.* SIGIR 2020 short. (**DC-BERT**)
### Incorporating Sparse Retrieval
- [Sparse, Dense, and Attentional Representations for Text Retrieval](https://arxiv.org/pdf/2005.00181.pdf) *Yi Luan et al.* TACL 2020. [[code](https://github.com/google-research/language/tree/master/language/multivec)] (**ME-BERT**)
- [The Curse of Dense Low-Dimensional Information Retrieval for Large Index Sizes.](https://arxiv.org/abs/2012.14210) *Nils Reimers et al.* ACL 2021.
- [COIL: Revisit Exact Lexical Match in Information Retrieval with Contextualized Inverted List.](https://arxiv.org/abs/2104.07186) *Luyu Gao et al.* NAACL 2021. (**COIL**)
- [Context-Aware Term Weighting For First Stage Passage Retrieval.](https://dl.acm.org/doi/pdf/10.1145/3397271.3401204) *Zhuyun Dai et al.* SIGIR 2020 short. [[code](https://github.com/AdeDZY/DeepCT)] (**DeepCT**)
- [Context-Aware Document Term Weighting for Ad-Hoc Search.](https://dl.acm.org/doi/pdf/10.1145/3366423.3380258) *Zhuyun Dai et al.* WWW 2020. [[code](https://github.com/AdeDZY/DeepCT/tree/master/HDCT)] (**HDCT**)
- [Learning Passage Impacts for Inverted Indexes.](https://arxiv.org/pdf/2104.12016.pdf) *Antonio Mallia et al.* SIGIR 2021 short. [[code](https://github.com/DI4IR/SIGIR2021)] (**DeepImapct**)
- [SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking.](https://dl.acm.org/doi/pdf/10.1145/3404835.3463098) *Thibault Formal et al.* SIGIR 2021.
- [Local Self-Attention over Long Text for Efficient Document Retrieval](https://arxiv.org/pdf/2005.04908v1.pdf) *Sebastian Hofstätter et al.* SIGIR 2020
- [BERT-based Dense Retrievers Require Interpolation
with BM25 for Effective Passage Retrieval.](https://espace.library.uq.edu.au/data/UQ_006a894/shuai2021interpolateDR.pdf?dsi_version=ed279d6cfbcc051a0067eb647d384567&Expires=1627966047&Key-Pair-Id=APKAJKNBJ4MJBJNC6NLQ&Signature=ICspaGP2y7Rm~H38mGz2YUeY7DleOyCemVm9zH6KPONEkNBPnSP-s39G9bTgsxlZvkXYBsy~W5YrJdWWcbmW7XT3MXAOA67W47ApvB2ov1eafpljmgBbDbBtSdyLqIEulP3Ty21t3blNgn6o1hXjJZvyAu19kDJvh-apnJ5~CaebiNYTtLZ72i4XMTUheP3EnEl1sMm92WYSDWX2-UsWT~9JTNF9Bf08S-xbTkl~oQ7dKdKWKyHJNv3TJVGmpvId-2uzxv57o4~ud69koPyNvvxrmtzEF5wSYIA6betv7prAItcU6M7vv36NHMr~9IuPsJYFW2yjG1jXt2IXM-Y8Ew__) *Shuai Wang et al.* ICTIR 2021.
- [Zero-shot Neural Passage Retrieval via Domain-targeted Synthetic Question Generation.](https://arxiv.org/pdf/2004.14503.pdf) *Ji Ma* Arxiv 2021.

## Dense Retrieval for Downstream Applications
### Question Answering
- [XOR QA: Cross-lingual Open-Retrieval Question Answering.](https://arxiv.org/abs/2010.11856) *Akari Asai et al.* NAACL 2021. [[code](https://nlp.cs.washington.edu/xorqa)]
- [Answering Complex Open-Domain Questions with Multi-Hop Dense Retrieval.](https://arxiv.org/pdf/2009.12756.pdf) *Wenhan Xiong at.el.* ICLR 2021 [[code](https://github.com/facebookresearch/multihop_dense_retrieval)]
- [Distilling Knowledge from Reader to Retriever for Question Answering.](https://openreview.net/pdf?id=NTEz-6wysdb) *Gautier Izacard, Edouard Grave* ICLR 2021. [[code](github.com/facebookresearch/FiD)]
- [A Replication Study of Dense Passage Retriever.](https://arxiv.org/pdf/2104.05740.pdf) *Xueguang Ma et al.* Arxiv 2021.
- [RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering.](https://arxiv.org/pdf/2010.08191.pdf) *Yingqi Qu et al.* NAACL 2021. (**RocketQA**)
- [Is Retriever Merely an Approximator of Reader?](https://arxiv.org/pdf/2010.10999.pdf) *Sohee Yang et al.* Arxiv 2020.
- [Dense Passage Retrieval for Open-Domain Question Answering.](https://arxiv.org/pdf/2004.04906.pdf) *Vladimir Karpukhin,Barlas Oguz et al.* EMNLP 2020 [[code](https://github.com/facebookresearch/DPR)] (**DPR**)
- [Unified Open-Domain Question Answering with Structured and Unstructured Knowledge.](https://arxiv.org/pdf/2012.14610.pdf) *Barlas Oguz et al.* Arxiv 2020.
- [Generation-Augmented Retrieval for Open-domain Question Answering.](https://arxiv.org/abs/2009.08553) *Yuning Mao et al.* ACL 2021.
- [Neural Retrieval for Question Answering with Cross-Attention Supervised Data Augmentation.](https://arxiv.org/abs/2009.13815) *Yinfei Yang et al.* Arxiv 2020.
- [Latent Retrieval for Weakly Supervised Open Domain Question Answering.](https://arxiv.org/pdf/1906.00300.pdf) *Kenton Lee et al.* ACL 2019. [[code](https://github.com/google-research/language/blob/master/language/orqa/README.md)] (**ORQA, ICT**)
- [REALM: Retrieval-Augmented Language Model Pre-Training.](https://arxiv.org/pdf/2002.08909.pdf) *Kelvin Guu, Kenton Lee et al.* ICML 2020. [[code](https://github.com/google-research/language/blob/master/language/realm/README.md)] (**REALM**)
- [Simple and Efficient ways to Improve REALM.](https://arxiv.org/abs/2104.08710.pdf) *Vidhisha Balachandran et al.* Arxiv 2021. (**REALM++**)
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.](https://arxiv.org/abs/2005.11401) *Patrick Lewis et al.* NIPS 2020. [[code](https://github.com/huggingface/transformers/blob/master/examples/rag/)] (**RAG**)
- [End-to-End Training of Neural Retrievers for Open-Domain Question Answering.](https://arxiv.org/abs/2101.00408) *Devendra Singh Sachan et al.* ACL 2021. [[code](https://github.com/NVIDIA/Megatron-LM)]
- [Real-Time Open-Domain Question Answering with Dense-Sparse Phrase Index.](https://arxiv.org/pdf/1906.05807.pdf) *Minjoon Seo,Jinhyuk Lee et al.* ACL 2019. [[code](https://github.com/uwnlp/denspi)] (**DENSPI**)
- [Contextualized Sparse Representations for Real-Time Open-Domain Question Answering.](https://arxiv.org/pdf/1911.02896.pdf) *Jinhyuk Lee, Minjoon Seo et al.* ACL 2020. [[code](https://github.com/jhyuklee/sparc)] (**SPARC, sparse vectors**)
- [Learning Dense Representations of Phrases at Scale.](https://arxiv.org/pdf/2012.12624.pdf) *Jinhyuk Lee, Danqi Chen et al.* Arxiv 2021. [[code](https://github.com/jhyuklee/DensePhrases)] (**DensePhrases**)
- [Analysing Dense Passage Retrieval for Multi-hop Question Answering.](https://arxiv.org/pdf/2106.08433.pdf) *Georgios Sidiropoulos et al.* Arxiv 2021. 

### Re-ranking
- [Sparse, Dense, and Attentional Representations for Text Retrieval](https://arxiv.org/pdf/2005.00181.pdf) *Yi Luan et al.* TACL 2020. [[code](https://github.com/google-research/language/tree/master/language/multivec)] (**ME-BERT**)
- [RepBERT: Contextualized Text Embeddings for First-Stage Retrieval.](https://arxiv.org/pdf/2006.15498.pdf) *Jingtao Zhan et al.* Arxiv 2020. [[code](https://github.com/jingtaozhan/RepBERT-Index)] (**RepBERT**)
- [Embedding-based retrieval in facebook search.](https://arxiv.org/abs/2006.11632v1) *Jui-Ting Huang et al.* KDD 2020.
- [Modularized Transfomer-based Ranking Framework.](https://arxiv.org/pdf/2004.13313.pdf) *Luyu Gao et al.* EMNLP 2020. [[code](https://github.com/luyug/MORES)] (**MORES**)
- [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT.](https://arxiv.org/pdf/2004.12832.pdf) *Omar Khattab et al.* SIGIR 2020. [[code](https://github.com/stanford-futuredata/ColBERT)] (**ColBERT**)

## Dataset
- [BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://arxiv.org/abs/2104.08663) *Nandan Thakur et al.* Arxiv 2021.
- [MS MARCO: A Human Generated MAchine Reading COmprehension Dataset](https://arxiv.org/pdf/1611.09268.pdf) *Payal Bajaj et al.* NIPS 2016.
- [Natural Questions: a Benchmark for Question Answering Research](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/1f7b46b5378d757553d3e92ead36bda2e4254244.pdf) *Tom Kwiatkowski et al.* TACL 2019.
- [TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension.](https://arxiv.org/abs/1705.03551) *Mandar Joshi et al.* ACL 2017.
- [Simple Entity-Centric Questions Challenge Dense Retrievers.](https://arxiv.org/pdf/2109.08535.pdf) *Christopher Sciavolino et al.* Arxiv 2021. [[code](https://github.com/princeton-nlp/EntityQuestions)]

## Other Resources


### Some Retrieval Toolkits
- [Faiss: a library for efficient similarity search and clustering of dense vectors](https://github.com/facebookresearch/faiss)
- [Pyserini: a Python Toolkit to Support Sparse and Dense Representations](https://github.com/castorini/pyserini/)
- [MatchZoo: a library consisting of many popular neural text matching models](https://github.com/NTMC-Community/MatchZoo)
- [Anserini: Enabling the Use of Lucene for Information Retrieval Research](https://github.com/castorini/anserini)

### Other Summaries
- [awesome-pretrained-models-for-information-retrieval](https://github.com/Albert-Ma/awesome-pretrained-models-for-information-retrieval)
