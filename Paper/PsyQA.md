# PsyQA

**Introduction:** a Chinese dataset of psychological health support in the form of Q&A.

**Property:** 

* the **question** along with a **detailed description** and several **keyword tags** is **posted by an anonymous help-seeker**, where the description generally **contains dense persona and emotion information** about the help-seeker.

* the answer is usually quite long (524 words on average), which is **replied asynchronously from the well-trained volunteers or professional counselors**, containing both the **detailed analysis** of the seeker's problem and the **guidance** for the seeker.

  * ==Q: How to use the dataset?==

* a portion of the answer are also additionally annotated by professional workers with typical support **strategies**, which are based on the psychological counseling theories.

  * The definition and example of different strategies in our guideline, together with the lexical features of the strategies in our annotated dataset.

  ![image-20210718154429210](.\imgs\image-20210718154429210.png)

**3 Distinct Characteristics:**

1. the corpus covers abundant mental health topics from 9 categories including emotion, relationships, and so on.
2. the answers in PsyQA are mostly provided by experienced and well-trained volunteers or professional counselors.
3. provide support strategy annotations for a portion of answers.
   * Different strategies in the answer are colored differently. Strategies **Information, Interpretation, Restatement, and Direct Guidance** are used in this answer.

* The **contextual information** greatly benefits the performance of support strategy identification.
* Experimental results also demonstrate that utilizing support strategies improves the answers
  generated by the models in terms of **their language fluency, coherence, and the ability to be on-topic and helpful.**

**Contribution**: 

1. We collect PsyQA, a high-quality Chinese dataset of psychological health support in the form of QA pair. The answers in PsyQA are usually long, which are provided by well-trained volunteers or professional counselors.

2. We annotate a portion of answer texts with a set of strategies for mental health support based on psychological counseling theories. Our analysis reveals that **there are not only typical lexical features in the texts of different strategies, but also explicit patterns of strategy organization and utilization in the answers.**
3. We conduct experiments of both **strategy identification** and **answer text generation** on PsyQA. Results demonstrate the importance of **using support strategies**, meanwhile indicating a large space for future research.

## Data Collection

Our dataset is crawled from [the Q&A column of Yixinli](xinli001.com/qa). Yixinli manually review and block unsafe contents to **avoid potential ethical risks** and ensure the quality of the data. We calculate that in our dataset, the help-supporters have ever **answered over 250 questions on average**. Besides, **8% answers** are from help-supporters who are **State-Certificated Class 2 Psychological Counselors**, and **35% answer**s are from **volunteers hired by Yixinli.**

* ==Q: What about the remained 57%?==

## Data Cleaning

* Remove personal information, duplicate line breaks, emojis, website links and advertisements by rule-based filtering.
* To ensure a higher quality, only those answers with more than 100 words were retained.
* Filter out questions that are not actually seeking for mental health support based on **keywords (topics) given by the poster**.

## Strategy Annotation and Quality Control

Multiple high-quality answers in our corpus and found that the strategies employed by the help-supporters are consistent with Helping Skills System (HSS).

* We assumed that a whole answer is **realized through an organized strategy sequence**, which may reveal the common layout of high-quality responses from mental health counselors.

We randomly sampled 4,012 questions (about 17.9%) in our dataset and picked their highest-voted answers.

Then we recruited and trained 9 workers to annotate the answers following our guideline. The workers **were required to read the guideline and the provided annotated examples before annotation**. To verify the effectiveness of training, we asked them to annotate 100 examples before formal annotation, which were revised by psychology professionals for feedback.

* repeated the above process until the workers were able to annotate the cases almost correctly.

After annotation, to check the quality of labels, we randomly sampled 200 annotated Q&A pairs, gave them to 2 examiners (both are graduate students of Clinical Psychology) to pick out incorrect labels, and calculated the consistency proportion.

## Strategy Identification

Present a strong sentence-level strategy identification model using RoBERTa.

### Data Preparation

Split into Train(80%), Dev(10%) and Test(10%).

### Model Architecture

Chinese RoBERTa base-version with 12 layers, adding **a dense output layer** on top of the pre-trained model with a cross-entropy loss function.

* For the model with contextual information, we input multiple consecutive sentences $S_1,S_2,S_3,\cdots$ to RoBERTa in the form of $[CLS]S_1[SEP][CLS]S_2[SEP][CLS]S_3$ and compute the mean loss of $[CLS]$ locating at the head of each sentence.
* For baseline model without contextual information, we input one sentence into RoBERTa and predict one sentence at one time.
* ==Need insight for RoBERTa==

### Result

![image-20210718162004050](.\imgs\image-20210718162004050.png)

Limited F1-score for Restatement and Information for reason:

1. No adding the Question into the input (due to the limitation of the maximum context length of RoBERTa) to help identify Restatement.

2. Extra psychological knowledge is needed to identify Information.
   * ==Q: a bit farfetched==

Conclusion:

1. Contextual information contains the inherent connection to the strategy sequence and the model recognizes the strategy patterns and performs better.
2. The gap between models and humans shows that this task is challenging and there is much room for future research.

## Answer Generation

**Given a triple** (question $S_Q$, description $S_D$, keyword set $K$) as input, where $S_Q$, $S_D$ are both sentences and $K$ are composed by at most 4 keywords, this task is to **generate a long counseling text** consisting of multiple sentences that could **give helpful comforts and advice mimicking a mental health counselor**.

## Model Pretraining

GPT-2 has shown its success on various language generation tasks. But

1. The pretrained Chinese GPT-2 available does not train on any corpus related to psychology or mental health support.
2. The context length of our dataset is more than 512, which existing small or middle size Chinese pretrained GPT-2 cannot deal with.

Thus we crawled 50K articles (0.1B tokens in total) related to psychology and mental health support from [Yixinli](xinli001.com/info) and train a GPT-2 from scratch based on the corpus.

* The maximum context length is 1,024 and the model contains 10 layers with 12 attention heads (resulting in 81.9M parameters)

* ==Q: why to train with another data set?==

### Implementation

#### Data Preparation

1. First predict the strategy of each sentence using our strategy classifier with contextual information.

2. Then mix the human annotated and classifier predicted parts of our dataset and randomly split them into train (90%), dev (5%), and test (5%) sets.

#### Prepending Strategy Token

To study the effectiveness of **using explicit strategy as input**, we compare the performance between models trained with/without strategy labels.

Formally, the prompt (model input) can be represented as
$$
[QUE]S_Q[DESC]S_D[KWD]K[ANS]
$$
Similarly, the goal text of the model with strategy labels can be represented as
$$
[\text{Strategy}_1]S_1[\text{Strategy}_2]S_2[\text{Strategy}_3]S_3\cdots
$$

#### Baseline Model

1. Seq2Seq model based on Transformer(S2S) with 5 layers encoder and 5 layers decoder
2. GPT-2 model only trained on PsyQA from scratch ($GPT_{sc}$)

#### Automatic Evaluation

Include:

* Perplexity
* BLEU
* Distinct-1
* Distinct-2
* CTRB: compute the consistency proportion between prediction and the strategy token locating at the head of the text spans

#### Human Evaluation

We calculated Krippendorff???s $\alpha$ ($K-\alpha$) (Krippendorff,
2011) to measure inter-rater consistency

* **Fluency**: whether the answer is fluent and grammatical.
* **Coherence**: whether the answer is logical and well organized.
* **Relevance**: whether the descriptions in the answer are relevant to the question. 
* **Helpfulness**: whether the interpretations and suggestions are suitable from the psychological counseling perspective.

Conclusion: **all the generated answers have relatively low scores** for

1. our generated answer is quite long (more than 500 words), increasing the probability of machine making mistakes;
2. the professional raters are pretty sensitive and cautious about the suggestions and analysis in the answer, especially concerning ethical risks.
