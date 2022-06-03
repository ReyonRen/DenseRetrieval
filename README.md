# Awesome Research for Dense Retrieval


> A curated list of awesome papers related to dense retrieval.



## Table of Contents

- [Survey paper](#survey-paper)
- [Architecture](#architecture) 
- [Training](#training)
  - [Formulation](#formulation)
  - [Negative Selection](#negative-selection)
  - [Data Augmentation](#data-augmentation)
  - [Pre-training](#pre-training)
- [Indexing](#indexing)
- [Interation with Re-ranking](#interation-with-re-ranking)
- [Advanced Topics](#advanced-topics)
- [Applications to Downstream Tasks](#applications-to-downstream-tasks)
  - [Question Answering](#question-answering)
  - [Entity Linking](#entity-linking)
  - [Dialog](#dialog)
  - [Retrieval-augmented Language Model](retrieval-augmented-language-model)
- [Dataset](#dataset)
- [Other Resources](#other-resources)
  - [Toolkits](#some-retrieval-toolkits)
  - [Other Summaries](#other-summaries)


 
## Survey Paper
| **Paper** | **Author** | **Venue** | **Code** |
| --- | --- | --- | --- |
| [Pretrained Transformers for Text Ranking: BERT and Beyond.](https://arxiv.org/abs/2010.06467) | Jimmy Lin et al. | Synthesis HLT 2021 | NA |
| [Semantic Models for the First-stage Retrieval: A Comprehensive Review.](https://arxiv.org/pdf/2103.04831.pdf) | 	Yinqiong Cai et al. | Arxiv 2021 | NA |
| [Pre-training Methods in Information Retrieval](https://arxiv.org/pdf/2111.13853) | Yixing Fan et al. | Arxiv 2021 | NA |
| []() |  |  |  |


## Architecture

| **Paper** | **Author** | **Venue**  | **Code** |
| --- | --- | --- | --- |
| [Poly-encoders: Architectures and pre-training strategies for fast and accurate multi-sentence scoring.](https://arxiv.org/pdf/1905.01969.pdf) | Samuel Humeau et al. | ICLR 2020 | [Python](https://github.com/facebookresearch/ParlAI/tree/master/projects/polyencoder) |
| [Sparse, Dense, and Attentional Representations for Text Retrieval.](https://arxiv.org/pdf/2005.00181.pdf) | Yi Luan et al. | <div style="width: 150pt">TACL 2021 | [Python](https://github.com/google-research/language/tree/master/language/multivec) |
| [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT.](https://arxiv.org/pdf/2004.12832.pdf) | Omar Khattab et al. | SIGIR 2020 | [Python](https://github.com/stanford-futuredata/ColBERT) |
| [Query Embedding Pruning for Dense Retrieval.](https://arxiv.org/pdf/2108.10341) | Nicola Tonellotto et al. | CIKM 2021 | [Python](https://github.com/terrierteam/pyterrier_colbert) |
| [DC-BERT: Decoupling Question and Document for Efficient Contextual Encoding.](https://arxiv.org/pdf/2002.12591.pdf) | Yuyu Zhang et al. | SIGIR 2020 | NA |
| [Real-Time Open-Domain Question Answering with Dense-Sparse Phrase Index.](https://arxiv.org/pdf/1906.05807.pdf) | Minjoon Seo et al. | ACL 2019 | [Python](https://github.com/uwnlp/denspi) |
| [Learning Dense Representations of Phrases at Scale.](https://arxiv.org/pdf/2012.12624.pdf) | Jinhyuk Lee et al. | ACL 2021 | [Python](https://github.com/jhyuklee/DensePhrases) |</div>
| [Phrase Retrieval Learns Passage Retrieval, Too. ](https://arxiv.org/pdf/2109.08133.pdf) | Jinhyuk Lee et al. | <div style="width: 150pt">EMNLP 2021</div> | [Python](https://github.com/princeton-nlp/DensePhrases.) |
| [Dense Hierarchical Retrieval for Open-Domain Question Answering.](https://arxiv.org/pdf/2110.15439) | Ye Liu et al. | EMNLP 2021 | [Python](https://github.com/yeliu918/DHR) |
| [The Curse of Dense Low-Dimensional Information Retrieval for Large Index Sizes.](https://arxiv.org/pdf/2012.14210) | Nils Reimers et al. | ACL 2021 |  NA |
| [Local Self-Attention over Long Text for Efficient Document Retrieval.](https://arxiv.org/pdf/2005.04908v1.pdf) | Sebastian Hofstätter et al. | SIGIR 2020 | [Python](https://github.com/sebastian-hofstaetter/transformer-kernel-ranking) |
| [Predicting Efficiency/Effectiveness Trade-offs for Dense vs. Sparse Retrieval Strategy Selection.](https://arxiv.org/pdf/2109.10739) | Negar Arabzadeh et al. | CIKM 2021 | [Python](https://github.com/Narabzad/Retrieval-Strategy-Selection.) |

## Training
### Formulation
| **Paper** | **Author** | **Venue** | **Code** |
| --- | --- | --- | --- |
| [More Robust Dense Retrieval with Contrastive Dual Learning. ](https://arxiv.org/pdf/2107.07773.pdf) | Yizhi Li et al. | ICTIR 2021 | [Python](https://github.com/thunlp/DANCE) |
| [PAIR: Leveraging Passage-Centric Similarity Relation for Improving Dense Passage Retrieval.](https://aclanthology.org/2021.findings-acl.191) | Ruiyang Ren et al. | ACL 2021 | [Python](https://github.com/PaddlePaddle/RocketQA/tree/main/research/PAIR_ACL2021) |
| [xMoCo: Cross Momentum Contrastive Learning for Open-Domain Question Answering.](https://aclanthology.org/2021.acl-long.477.pdf) | Nan Yang et al. |  ACL 2021 | NA |
| [A Modern Perspective on Query Likelihood with Deep Generative Retrieval Models.](https://arxiv.org/pdf/2106.13618) | Oleg Lesota et al. | ICTIR 2021 | [Python]() |

### Negative Selection
| **Paper** | **Author** | **Venue** | **Code** |
| --- | --- | --- | --- |
| [Learning To Retrieve: How to Train a Dense Retrieval Model Effectively and Efficiently.](https://arxiv.org/abs/2010.10469) | Jingtao Zhan et al. | Arxiv 2020 | NA |
| [Dense Passage Retrieval for Open-Domain Question Answering.](https://arxiv.org/pdf/2004.04906.pdf) | Vladimir Karpukhin et al. | EMNLP 2020 | [Python](https://github.com/facebookresearch/DPR) |
| [RepBERT: Contextualized Text Embeddings for First-Stage Retrieval.](https://arxiv.org/pdf/2006.15498.pdf) | Jingtao Zhan et al | Arxiv 2020 | [Python](https://github.com/jingtaozhan/RepBERT-Index) |
| [Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval.](https://arxiv.org/pdf/2007.00808.pdf) | Lee Xiong et al. | ICLR 2021 | [Python](https://github.com/microsoft/ANCE) |
| [Optimizing Dense Retrieval Model Training with Hard Negatives.](https://arxiv.org/abs/2104.08051) | Jingtao Zhan et al | SIGIR 2021 | [Python](https://github.com/jingtaozhan/DRhard) |
| [Neural Passage Retrieval with Improved Negative Contrast.](https://arxiv.org/abs/2010.12523) | Jing Lu et al. | Arxiv 2020 | NA |
| [RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering.](https://arxiv.org/pdf/2010.08191.pdf) | Yingqi Qu et al. | NAACL 2021 | [Python](https://github.com/PaddlePaddle/RocketQA/tree/main/research/RocketQA_NAACL2021) |
| [Efficiently Teaching an Effective Dense Retriever with Balanced Topic Aware Sampling.](https://arxiv.org/pdf/2104.06967.pdf) | Sebastian Hofstätter et al. | SIGIR 2021 | [Python](https://github.com/sebastian-hofstaetter/tas-balanced-dense-retrieval) |
| [Scaling deep contrastive learning batch size under memory limited setup](https://arxiv.org/pdf/2101.06983v2) | Luyu Gao et al. | RepL4NLP 2021 | [Python](https://github.com/luyug/GradCache) |
| [Multi-stage training with improved negative contrast for neural passage retrieval.](https://aclanthology.org/2021.emnlp-main.492.pdf) | Jing Lu et al. | EMNLP 2021 | NA |
| [Learning robust dense retrieval models from incomplete relevance labels](https://dl.acm.org/doi/pdf/10.1145/3404835.3463106) | Prafull Prakash et al. | SIGIR 2021 | [Python](https://github.com/purble/RANCE) |
| [Efficient Training of Retrieval Models Using Negative Cache](https://papers.nips.cc/paper/2021/file/2175f8c5cd9604f6b1e576b252d4c86e-Paper.pdf) | Erik M. Lindgren et al. | NeurIPS 2021 | [Python](NA) |
| [CODER: An efficient framework for improving retrieval through COntextual Document Embedding Reranking](https://arxiv.org/pdf/2112.08766) | George Zerveas et al. | Arxiv 2021 | NA |

### Data Augmentation
| **Paper** | **Author** | **Venue** | **Code** |
| --- | --- | --- | --- |
| [Unified Open-domain Question Answering with Structured and Unstructured Knowledge.](https://arxiv.org/pdf/2012.14610v1.pdf) | Barlas Oguz et al. | Arxiv 2021 | NA |
| [Augmented SBERT: Data Augmentation Method for Improving Bi-Encoders for Pairwise Sentence Scoring Tasks.](https://arxiv.org/abs/2010.08240) | Nandan Thakur et al. | NAACL 2021 | [Python](www.sbert.net) |
| [Is Retriever Merely an Approximator of Reader?](https://arxiv.org/pdf/2010.10999.pdf) | Sohee Yang et al. | Arxiv 2020 | [Python]() |
| [Distilling Knowledge from Reader to Retriever for Question Answering.](https://openreview.net/pdf?id=NTEz-6wysdb) | Gautier Izacard et al. | ICLR 2021 | [Python](github.com/facebookresearch/FiD) |
| [Distilling Knowledge for Fast Retrieval-based Chat-bots.](https://arxiv.org/pdf/2004.11045.pdf) | Amir Vakili Tahami et al. | SIGIR 2020 | [Python](https://github.com/KamyarGhajar/DistilledNeuralResponseRanker) |
| [Improving Efficient Neural Ranking Models with Cross-Architecture Knowledge Distillation.](https://arxiv.org/pdf/2010.02666.pdf) | Sebastian Hofstätter et al. | Arxiv 2020 | [Python](https://github.com/sebastian-hofstaetter/neural-ranking-kd) |
| [Distilling Dense Representations for Ranking using Tightly-Coupled Teachers.](https://arxiv.org/pdf/2010.11386.pdf) | Sheng-Chieh Lin et al. | Arxiv 2020 | [Python](https://github.com/castorini/pyserini/blob/master/docs/experiments-tctcolbert) |
| [In-Batch Negatives for Knowledge Distillation with Tightly-Coupled Teachers for Dense Retrieval.](https://aclanthology.org/2021.repl4nlp-1.17/) | Sheng-Chieh Lin et al. | RepL4NLP 2021 | [Python](https://github.com/castorini/pyserini/blob/master/docs/experiments-tct_colbert-v2.md) |
| [Neural Retrieval for Question Answering with Cross-Attention Supervised Data Augmentation.](https://arxiv.org/abs/2009.13815) | Yinfei Yang et al. | ACL 2021 | NA |
| [Enhancing Dual-Encoders with Question and Answer Cross-Embeddings for Answer Retrieval.](https://aclanthology.org/2021.findings-emnlp.198.pdf) | Yanmeng Wang et al. | EMNLP 2021 | NA |
| [Pseudo Label based Contrastive Sampling for Long Text Retrieval](https://ieeexplore.ieee.org/abstract/document/9675219) | Le Zhu et al. | IALP 2021 | NA |
| [Multi-View Document Representation Learning for Open-Domain Dense Retrieval](https://arxiv.org/pdf/2203.08372.pdf) | Shunyu Zhang et al. | ACL 2022 | NA |
| [Augmenting Document Representations for Dense Retrieval with Interpolation and Perturbation](https://arxiv.org/pdf/2203.07735v2.pdf) | Soyeong Jeong et al. | ACL 2022 | [Python](github.com/starsuzi/DAR) |

### Pre-training
| **Paper** | **Author** | **Venue** | **Code** |
| --- | --- | --- | --- |
| [Latent Retrieval for Weakly Supervised Open Domain Question Answering.](https://arxiv.org/pdf/1906.00300.pdf) | Kenton Lee et al. | ACL 2019 | [Python](https://github.com/google-research/language/blob/master/language/orqa/README.md) |
| [Pre-training tasks for embedding-based large scale retrieval.](https://arxiv.org/pdf/2002.03932.pdf) | Wei-Cheng Chang et al. | ICLR 2020 | NA |
| [REALM: Retrieval-Augmented Language Model Pre-Training.](https://arxiv.org/pdf/2002.08909.pdf) | Kelvin Guu et al. | ICML 2020 | [Python](https://github.com/google-research/language/blob/master/language/realm/README.md) |
| [PROP: Pre-training with Representative Words Prediction for Ad-hoc Retrieval.](https://arxiv.org/pdf/2010.10137.pdf) | Xinyu Ma et.al. | WSDM 2021 | [Python](https://github.com/Albert-Ma/PROP) |
| [B-PROP: Bootstrapped Pre-training with Representative Words Prediction for Ad-hoc Retrieval.](https://arxiv.org/pdf/2104.09791.pdf) | Xinyu Ma et al. | SIGIR 2021 | NA |
| [Domain-matched Pre-training Tasks for Dense Retrieval.](https://arxiv.org/pdf/2107.13602.pdf) | Barlas Oguz et al. | Arxiv 2021 | [Python]() |
| [Less is More: Pre-train a Strong Text Encoder for Dense Retrieval Using a Weak Decoder.](https://arxiv.org/pdf/2102.09206.pdf) | Shuqi Lu et al. | EMNLP 2021 | [Python]() |
| [Sentence-T5: Scalable Sentence Encoders from Pre-trained Text-to-Text Models](https://arxiv.org/pdf/2108.08877) | Jianmo Ni et al. | Arxiv 2021 | [Python](https://github.com/google-research/text-to-text-transfer-transformer) |
| [Unsupervised Corpus Aware Language Model Pre-training for Dense Passage Retrieval.](https://arxiv.org/pdf/2108.05540.pdf) | Luyu Gao et al. | ACL 2022 | [Python](https://github.com/luyug/Condenser) |
| [Condenser: a Pre-training Architecture for Dense Retrieval](https://arxiv.org/pdf/2104.08253.pdf) | Luyu Gao et al. | EMNLP 2021 | [Python](https://github.com/luyug/Condenser) |
| [TSDAE: Using Transformer-based Sequential Denoising Auto-Encoder for Unsupervised Sentence Embedding Learning.](https://arxiv.org/pdf/2104.06979) | Kexin Wang et al. | EMNLP 2021 | [Python](https://github.com/UKPLab/sentence-transformers/) |
| [SimCSE: Simple Contrastive Learning of Sentence Embeddings.](https://arxiv.org/pdf/2104.08821.pdf) | Tianyu Gao et al. | EMNLP 2021 | [Python](https://github.com/princeton-nlp/SimCSE) |
| [Semantic Re-Tuning With Contrastive Tension. ](https://openreview.net/pdf?id=Ov_sMNau-PF) | Fredrik Carlsson et al. | ICLR 2021 | [Python](https://github.com/FreddeFrallan/Contrastive-Tension) |
| [Simple and Efficient ways to Improve REALM](https://arxiv.org/abs/2104.08710.pdf) | Vidhisha Balachandran et al. | Arxiv 2021 | NA |
| [Towards Robust Neural Retrieval Models with Synthetic Pre-Training.](https://arxiv.org/pdf/2104.07800v1) | Revanth Gangi Reddy et al. | Arxiv 2021 | NA |
| [Pre-training for ad-hoc retrieval: Hyperlink is also you need.](https://arxiv.org/pdf/2108.09346) | Zhengyi Ma et al. | CIKM 2021 | [Python](https://github.com/zhengyima/anchors) |
| [Hyperlink-induced Pre-training for Passage Retrieval in Open-domain Question Answering.]() | Jiawei Zhou et al. | ACL 2022 | [Python](https://github.com/jzhoubu/HLP) |


## Indexing
| **Paper** | **Author** | **Venue** | **Code** |
| --- | --- | --- | --- |
| [Context-Aware Term Weighting For First Stage Passage Retrieval.](https://dl.acm.org/doi/pdf/10.1145/3397271.3401204) | Zhuyun Dai et al. | SIGIR 2020 | [Python](https://github.com/AdeDZY/DeepCT) |
| [Context-Aware Document Term Weighting for Ad-Hoc Search.](https://dl.acm.org/doi/pdf/10.1145/3366423.3380258) | Zhuyun Dai et al. | WWW 2020 | [Python](https://github.com/AdeDZY/DeepCT/tree/master/HDCT) |
| [COIL: Revisit Exact Lexical Match in Information Retrieval with Contextualized Inverted List.](https://arxiv.org/abs/2104.07186) | Luyu Gao et al. | NAACL 2021 | [Python](https://github.com/luyug/COIL) |
| [Learning Passage Impacts for Inverted Indexes.](https://arxiv.org/pdf/2104.12016.pdf) | Antonio Mallia et al. | SIGIR 2021 | [Python](https://github.com/DI4IR/SIGIR2021) |
| [SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking.](https://arxiv.org/pdf/2107.05720) | Thibault Formal et al. | SIGIR 2021 | [Python](https://github.com/naver/splade) |
| [SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval.](https://arxiv.org/pdf/2109.10086.pdf) | Thibault Formal et al. | Arxiv 2021 | [Python](https://github.com/naver/splade) |
| [BERT-based Dense Retrievers Require Interpolation with BM25 for Effective Passage Retrieval.](https://arvinzhuang.github.io/files/shuai2021interpolateDR.pdf) | Shuai Wang et al. | ICTIR 2021 | [Python](https://github.com/ielab/InterpolateDR-ICTIR2021) |
| [Predicting Efficiency/Effectiveness Trade-offs for Dense vs. Sparse Retrieval Strategy Selection.](https://arxiv.org/pdf/2109.10739.pdf) | Negar Arabzadeh et al. | CIKM 2021 | [Python](https://github.com/Narabzad/Retrieval-Strategy-Selection) |
| [Accelerating Large-Scale Inference with Anisotropic Vector Quantization.](https://arxiv.org/pdf/1908.10396) | Ruiqi Guo et al. | Arxiv 2019 | [Python](http://ann-benchmarks.com/) |
| [Jointly Optimizing Query Encoder and Product Quantization to Improve Retrieval Performance.](https://arxiv.org/pdf/2108.00644.pdf) | Jingtao Zhan et al. | CIKM 2021 | [Python](https://github.com/jingtaozhan/JPQ) |
| [Learning Discrete Representations via Constrained Clustering for Effective and Efficient Dense Retrieval.](https://www.semanticscholar.org/paper/Learning-Discrete-Representations-via-Constrained-Zhan-Mao/91429255eefe48ad140ccfaf6aa1e6be11a72a53) | Jingtao Zhan et al. | WSDM 2022 | [Python](https://github.com/jingtaozhan/repconc) |
| [Joint Learning of Deep Retrieval Model and Product Quantization based Embedding Index.](https://arxiv.org/pdf/2105.03933) | Han Zhang et al. | SIGIR 2021 | [Python](https://github.com/jdcomsearch/poeem) |
| [Efficient Passage Retrieval with Hashing for Open-domain Question Answering.](https://arxiv.org/pdf/2106.00882) | Ikuya Yamada et al. | ACL 2021 | [Python](https://github.com/studio-ousia/bpr) |
| [A Memory Efficient Baseline for Open Domain Question Answering](https://arxiv.org/pdf/2012.15156.pdf) | Gautier Izacard et al. | Arxiv 2020 | NA |
| [Simple and Effective Unsupervised Redundancy Elimination to Compress Dense Vectors for Passage Retrieval.](https://cs.uwaterloo.ca/~jimmylin/publications/Ma_etal_EMNLP2021.pdf) | Xueguang Ma et al. | EMNLP 2021 | [Python](http://pyserini.io/) |
| [The Curse of Dense Low-Dimensional Information Retrieval for Large Index Sizes.](https://arxiv.org/pdf/2012.14210.pdf) | Nils Reimers et al. | ACL 2021 | NA |
| [Matching-oriented Product Quantization For Ad-hoc Retrieval.](https://arxiv.org/pdf/2104.07858) | Shitao Xiao | EMNLP 2021 | [Python](https://github.com/microsoft/MoPQ) |

## Interation with Re-ranking
| **Paper** | **Author** | **Venue** | **Code** |
| --- | --- | --- | --- |
| [RocketQAv2: A Joint Training Method for Dense Passage Retrieval and Passage Re-ranking](https://aclanthology.org/2021.emnlp-main.224.pdf) | Ruiyang Ren et al. | EMNLP 2021 | [Python](https://github.com/PaddlePaddle/RocketQA/tree/main/research/RocketQAv2_EMNLP2021) |
| [Dealing with Typos for BERT-based Passage Retrieval and Ranking.](https://arxiv.org/pdf/2108.12139.pdf) | Shengyao Zhuang et al. | EMNLP 2021 | [Python](https://github.com/ielab/typos-aware-BERT) |
| [Trans-Encoder: Unsupervised sentence-pair modelling through self- and mutual-distillations.](https://arxiv.org/pdf/2109.13059.pdf) | Fangyu Liu et al. | ICLR 2022 | [Python](https://github.com/amzn/trans-encoder) |
| [Adversarial Retriever-Ranker for dense text retrieval.](https://arxiv.org/pdf/2110.03611.pdf) | Hang Zhang | Arxiv 2021 | NA |
| [Embedding-based Retrieval in Facebook Search.](https://dl.acm.org/doi/pdf/10.1145/3394486.3403305) | Jui-Ting Huang et al. | KDD 2020 | NA |


## Advanced Topics
### Axiomatic Analysis
| **Paper** | **Author** | **Venue** | **Code** |
| --- | --- | --- | --- |
| [DiffIR: Exploring Differences in Ranking Models’ Behavior.](https://dl.acm.org/doi/pdf/10.1145/3404835.3462784) | Kevin Martin Jose et al. | SIGIR 2021 | [Python](https://github.com/capreolus-ir/diffir) |
| [A White Box Analysis of ColBERT.](https://arxiv.org/pdf/2012.09650) | Thibault Formal et al. | ECIR 2021 | NA |
| [Towards Axiomatic Explanations for Neural Ranking Models.](https://arxiv.org/pdf/2106.08019) | Michael Völske et al. | ICTIR 2021 | [Python](https://github.com/webis-de/ICTIR-21) |
| [ABNIRML: Analyzing the Behavior of Neural IR Models](https://arxiv.org/pdf/2011.00696.pdf) | Sean MacAvaney et al. | Arxiv 2020 | [Python](https://github.com/allenai/abnriml) |
| [Diagnosing BERT with Retrieval Heuristics.](https://arxiv.org/pdf/2201.04458.pdf) | Arthur Camara et al. | ECIR 2020 | NA |
| [How Does BERT Rerank Passages? An Attribution Analysis with Information Bottlenecks](https://aclanthology.org/2021.blackboxnlp-1.39.pdf) | Zhiying Jiang et al. | BlackboxNLP 2021 | NA |

### Generative Text Retrieval
| **Paper** | **Author** | **Venue** | **Code** |
| --- | --- | --- | --- |
| [Transformer Memory as a Diﬀerentiable Search Index.](https://arxiv.org/pdf/2202.06991.pdf) | Yi Tay et al. | Arxiv 2022 | NA |
| [DynamicRetriever: A Pre-training Model-based IR System with Neither Sparse nor Dense Index.](https://arxiv.org/pdf/2203.00537) | Yujia Zhou et al. | Arxiv 2022 | NA |
| [Autoregressive Search Engines: Generating Substrings as Document Identifiers.](https://arxiv.org/pdf/2204.10628.pdf) | Michele Bevilacqua et al. | Arxiv 2022 | [Python](https://github.com/facebookresearch/SEAL) |
| [Generative Retrieval for Long Sequences.](https://arxiv.org/pdf/2204.13596.pdf) | Hyunji Lee et al. | Arxiv 2022 | NA |
| [GERE: Generative Evidence Retrieval for Fact Verification](https://arxiv.org/pdf/2204.05511.pdf) | Jiangui Chen et al. | SIGIR 2022 | [Python](https://github.com/Chriskuei/GERE) |
| [Autoregressive Entity Retrieval.](https://arxiv.org/abs/2010.00904) | Nicola De Cao et al. | ICLR 2021 | [Python](https://github.com/facebookresearch/GENRE) |

### Zero-shot Dense Retrieval
| **Paper** | **Author** | **Venue** | **Code** |
| --- | --- | --- | --- |
| [BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models.](https://arxiv.org/abs/2104.08663) | Nandan Thakur et al. | NIPS 2021 | [Python](https://github.com/UKPLab/beir) |
| [A Thorough Examination on Zero-shot Dense Retrieval.](https://arxiv.org/pdf/2204.12755.pdf) | Ruiyang Ren et al. | Arxiv 2022 | NA |
| [Challenges in Generalization in Open Domain Question Answering.](https://arxiv.org/pdf/2109.01156.pdf) | Linqing Liu et al. | NAACL 2022 | [Python](https://github.com/likicode/QA-generalize) |
| [Zero-shot Neural Passage Retrieval via Domain-targeted Synthetic Question Generation.](https://arxiv.org/pdf/2004.14503.pdf) | Ji Ma et al. | Arxiv 2021 | NA |
| [Efficient Retrieval Optimized Multi-task Learning.](https://arxiv.org/pdf/2104.10129) | Hengxin Fun et al. | Arxiv 2021 | NA |
| [Zero-Shot Dense Retrieval with Momentum Adversarial Domain Invariant Representations.](https://arxiv.org/pdf/2110.07581.pdf) | Ji Xin et al. | ACL 2022 | NA |
| [Towards Robust Neural Retrieval Models with Synthetic Pre-Training.](https://arxiv.org/pdf/2104.07800v1.pdf) | Revanth Gangi Reddy et al. | Arxiv 2021 | NA |
| [Embedding-based Zero-shot Retrieval through Query Generation.](https://arxiv.org/pdf/2009.10270.pdf) | Davis Liang | Arxiv 2020 | NA |
| [GPL: Generative Pseudo Labeling for Unsupervised Domain Adaptation of Dense Retrieval.](https://arxiv.org/pdf/2112.07577.pdf) | Kexin Wang et al. | Arxiv 2021 | [Python](https://github.com/UKPLab/gpl) |
| [Salient Phrase Aware Dense Retrieval: Can a Dense Retriever Imitate a Sparse One?](https://arxiv.org/pdf/2110.06918.pdf) | Xilun Chen | Arxiv 2021 | [Python](https://github.com/facebookresearch/dpr-scale/tree/main/spar) |
| [LaPraDoR: Unsupervised Pretrained Dense Retriever for Zero-Shot Text Retrieval.](https://arxiv.org/pdf/2203.06169v2.pdf) | Canwen Xu et al. | ACL 2022 | [Python](https://github.com/JetRunner/LaPraDoR) |
| [Out-of-Domain Semantics to the Rescue! Zero-Shot Hybrid Retrieval Models.](https://arxiv.org/pdf/2201.10582.pdf) | Tao Chen | ECIR 2022 | NA |
| [Towards Unsupervised Dense Information Retrieval with Contrastive Learning.](https://arxiv.org/pdf/2112.09118v1.pdf) | Gautier Izacard et al. | Arxiv 2021 | NA |
| [Large Dual Encoders Are Generalizable Retrievers.](https://arxiv.org/pdf/2112.07899.pdf) | Jianmo Ni et al. | Arxiv 2021 | NA |

### Other Retrieval Settings
| **Paper** | **Author** | **Venue** | **Code** |
| --- | --- | --- | --- |
| [Open Domain Question Answering over Tables via Dense Retrieval](https://aclanthology.org/2021.naacl-main.43.pdf) | Jonathan Herzig et al. | NAACL 2021 | [Python](https://github.com/google-research/tapas.) |
| [Multi-modal Retrieval of Tables and Texts Using Tri-encoder Models.](https://arxiv.org/pdf/2108.04049v1.pdf) | Bogdan Kostic et al. | Arxiv 2021 | NA |
| [XOR QA: Cross-lingual Open-Retrieval Question Answering](https://aclanthology.org/2021.naacl-main.46.pdf) | Akari Asai et al. | NAACL 2021 | [Python](https://nlp.cs.washington.edu/xorqa/) |
| [One Question Answering Model for Many Languages with Cross-lingual Dense Passage Retrieval.](https://arxiv.org/pdf/2107.11976.pdf) | Akari Asai et al. | NeurIPS 2021 | [Python](https://github.com/AkariAsai/CORA) |

### Industrial Practice
| **Paper** | **Author** | **Venue** | **Code** |
| --- | --- | --- | --- |
| [Pre-trained Language Model for Web-scale Retrieval in Baidu Search.](https://arxiv.org/pdf/2106.03373v3) | Yiding Liu et al. | KDD 2021 | NA |
| [MOBIUS: Towards the Next Generation of Query-Ad Matching in Baidu’s Sponsored Search](http://research.baidu.com/Public/uploads/5d12eca098d40.pdf) | Miao Fan | KDD 2019 | NA |
| [Uni-Retriever: Towards Learning The Unified Embedding Based Retriever in Bing Sponsored Search.](https://arxiv.org/pdf/2202.06212.pdf) | Jianjin Zhang et al. | Arxiv 2022 | NA |
| [Embedding-based Product Retrieval in Taobao Search.](https://arxiv.org/pdf/2106.09297.pdf) | Sen Li et al. | KDD 2021 | NA |
| [Que2Search: Fast and Accurate Query and Document Understanding for Search at Facebook.](https://scontent-nrt1-1.xx.fbcdn.net/v/t39.8562-6/246795273_2109661252514735_2459553109378891559_n.pdf?_nc_cat=105&ccb=1-7&_nc_sid=ad8a9d&_nc_ohc=7LLAz1SvhvcAX9Dr2-E&_nc_ht=scontent-nrt1-1.xx&oh=00_AT_sJBUEVm6mlAYngNn31Oc2BTqokLB9dvcdHTLYsIDCqA&oe=629847E3) | Yiqun Liu et al. | KDD 2021 | NA |
| [DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node.](https://suhasjs.github.io/files/diskann_neurips19.pdf) | Suhas Jayaram Subramanya et al. | NeurIPS 2019 | [Python](https://github.com/Microsoft/DiskANN) |
| [SPANN: Highly-efficient Billion-scale Approximate Nearest Neighbor Search.](https://arxiv.org/pdf/2111.08566.pdf) | Qi Chen et al. | NeurIPS 2021 | [Python](https://github.com/microsoft/SPTAG) |


## Applications to Downstream Tasks
| **Paper** | **Author** | **Venue** | **Code** |
| --- | --- | --- | --- |
| [Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering.](https://aclanthology.org/2021.eacl-main.74.pdf) | Gautier Izacard et al. | ECAL 2021 | [Python](https://github.com/facebookresearch/FiD) |
| [Baleen: Robust Multi-Hop Reasoning at Scale via Condensed Retrieval.](https://arxiv.org/pdf/2101.00436.pdf) | Omar Khattab et al. | NeurIPS 2021 | [Python](https://github.com/stanford-futuredata/Baleen) |
| [You only need one model for open-domain question answering.](https://arxiv.org/pdf/2112.07381.pdf) | Haejun Lee et al. | Arxiv 2021 | NA |
| [Zero-Shot Entity Linking by Reading Entity Descriptions.](https://arxiv.org/pdf/1906.07348.pdf) | Lajanugen Logeswaran et al. | ACL 2019 | [Python](https://github.com/lajanugen/zeshel) |
| [Learning Dense Representations for Entity Retrieval.](https://aclanthology.org/K19-1049.pdf) | Daniel Gillick et al. | CoNLL 2019 | NA |
| [Scalable Zero-shot Entity Linking with Dense Entity Retrieval.](https://arxiv.org/pdf/1911.03814.pdf) | Ledell Wu et al. | EMNLP 2020 | [Python](https://github.com/facebookresearch/BLINK) |
| [Retrieval Augmentation Reduces Hallucination in Conversation.](https://arxiv.org/pdf/2104.07567.pdf) | Kurt Shuster et al. | EMNLP 2021 | NA |
| [PLATO-KAG: Unsupervised Knowledge-Grounded Conversation.](https://aclanthology.org/2021.nlp4convai-1.14.pdf) | Xinxian Huang et al. | NLP4ConvAI 2021 | [Python](https://github.com/PaddlePaddle/Knover/tree/develop/projects/PLATO-KAG) |
| [Internet-Augmented Dialogue Generation.](https://arxiv.org/pdf/2107.07566) | Mojtaba Komeili et al. | ACL 2022 | NA |
| [LaMDA: Language Models for Dialog Applications.](https://arxiv.org/pdf/2201.08239.pdf) | Romal Thoppilan et al. | Arxiv 2022 | NA |

## Dataset
| **Paper** | **Author** | **Venue** | **Link** |
| --- | --- | --- | --- |
| [BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://arxiv.org/abs/2104.08663) | Nandan Thakur et al. | NeurIPS 2021 | [Python]() |
| [MS MARCO: A Human Generated MAchine Reading COmprehension Dataset](https://arxiv.org/pdf/1611.09268.pdf) | Payal Bajaj et al. | NeurIPS 2016 | [Python]() |
| [Natural Questions: a Benchmark for Question Answering Research](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/1f7b46b5378d757553d3e92ead36bda2e4254244.pdf) | Tom Kwiatkowski et al. | TACL 2019 | [Python]() |
| [TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension.](https://arxiv.org/abs/1705.03551) | Mandar Joshi et al. | ACL 2017 | [Python]() |
| []() |  |  | [Python]() |
| []() |  |  | [Python]() |
| []() |  |  | [Python]() |
| []() |  |  | [Python]() |
| [Simple Entity-Centric Questions Challenge Dense Retrievers.](https://arxiv.org/pdf/2109.08535.pdf) | Christopher Sciavolino et al. | EMNLP 2021 | [Python]() |
| []() |  |  | [Python]() |
| []() |  |  | [Python]() |
| []() |  |  | [Python]() |
| []() |  |  | [Python]() |
| []() |  |  | [Python]() |
| []() |  |  | [Python]() |
| [ArchivalQA: A Large-scale Benchmark Dataset for Open Domain Question Answering over Archival News Collections.](https://arxiv.org/pdf/2109.03438v2.pdf) | Jiexin Wang et al. | Arxiv 2021 | NA |
| [SituatedQA: Incorporating Extra-Linguistic Contexts into QA](https://arxiv.org/pdf/2109.06157.pdf) | Michael J.Q. Zhang et al. | EMNLP 2021 | [Python](https://situatedqa.github.io/) |

## Other Resources
### Toolkits
| **Paper** | **Author** | **Venue** | **Code** |
| --- | --- | --- | --- |
| []() |  |  | [Python]() |
| []() |  |  | [Python]() |

### Other Summaries
| **Paper** | **Author** | **Venue** | **Code** |
| --- | --- | --- | --- |
| []() |  |  | [Python]() |
| []() |  |  | [Python]() |
<!-- 
](#interation-with-re-ranking)
- [Advanced Topics](#advanced-topics)
- [Applications to Downstream Tasks](#applications-to-downstream-tasks)
  - [Question Answering](#question-answering)
  - [Entity Linking](#entity-linking)
  - [Dialog](#dialog)
  - [Retrieval-augmented Language Model](retrieval-augmented-language-model)
- [Dataset](#dataset)
- [Other Resources](#other-resources)
  - [Toolkits](#some-retrieval-toolkits)
  - [Other Summaries](#other-summaries)


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
- [Neural Retrieval for Question Answering with Cross-Attention Supervised Data Augmentation.](https://arxiv.org/abs/2009.13815) *Yinfei Yang et al.* ACL 2021.
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
- [Predicting Efficiency/Effectiveness Trade-offs for Dense vs. Sparse Retrieval Strategy Selection.](https://arxiv.org/abs/2109.10739) *Negar Arabzadeh et al.* Arxiv 2021

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
- [Simple Entity-Centric Questions Challenge Dense Retrievers.](https://arxiv.org/pdf/2109.08535.pdf) *Christopher Sciavolino et al.* EMNLP 2021. [[code](https://github.com/princeton-nlp/EntityQuestions)]

## Other Resources


### Some Retrieval Toolkits
- [Faiss: a library for efficient similarity search and clustering of dense vectors](https://github.com/facebookresearch/faiss)
- [Pyserini: a Python Toolkit to Support Sparse and Dense Representations](https://github.com/castorini/pyserini/)
- [MatchZoo: a library consisting of many popular neural text matching models](https://github.com/NTMC-Community/MatchZoo)
- [Anserini: Enabling the Use of Lucene for Information Retrieval Research](https://github.com/castorini/anserini)

### Other Summaries
- [awesome-pretrained-models-for-information-retrieval](https://github.com/Albert-Ma/awesome-pretrained-models-for-information-retrieval) -->
