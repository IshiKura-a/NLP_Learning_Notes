# S1 Positive and Negative Comments

## Introduction

1. Represent texts as a vector and build a classifier by **Logistic Regression**.
2. Use **Naive Bayes** on the same problem.
3. **Vector Space Model**

## CH1 Logistic Regression

Supervised ML (training):

![image-20210712142955828](.\imgs\image-20210712142955828.png)

This chapter's target: use **logistic regression** to classify whether a tweet is positive or not, i.e. to do **sentiment analysis**.

![image-20210712143308071](.\imgs.\imgs\image-20210712143308071.png)

The first step is to represent a tweet into a vector.

First, build a frequency dictionary like this:

![image-20210712144639768](.\imgs.\imgs\image-20210712144639768.png)

It's build by the positive tweets and negative tweets separately.

Next, we conduct the feature extraction, by represent the $m^{\text{th}}$ tweet with the vector:
$$
X_m=[1, \sum_{w}\text{freq}(w,1), \sum_{w}\text{freq}(w,0)]
$$
where 1 is a bias item,  $\sum_{w}\text{freq}(w,1)$ is the sum of the positive frequency of all the words in the tweet and $\sum_{w}\text{freq}(w,0)$ is the sum of the negative frequency of all the words in the tweet.

For example, the tweet "I am sad, I am not learning NLP" is represent as $[1,8,10]$, where 8 = 3+3+1+1, and 11 = 3+3+1+1+1+1.

From examples listed above, we found the word "am" got a quite high frequency, while the word per se seems to have no influence on the classification. Thus, before we represent the tweet into a vector, we need to do some preprocessing, for example **remove the words that are not that important.**

Basically, we have **Stop words** and **Punctuation** e.g.

![image-20210712150407404](.\imgs.\imgs\image-20210712150407404.png)

Meanwhile, a tweet usually have **Handle**s and **URL**s, which also have no influence on sentiment analysis and can be removed.

![image-20210712151213231](.\imgs.\imgs\image-20210712151213231.png)

Also, a word may have multiple formats like word "tune", may be "tuning", "tunes", "TUNE" and so on. To address this problem, we use **Stemming** and **Lowercase**.

![image-20210712151821949](.\imgs.\imgs\image-20210712151821949.png)

Now, we get the vectors of input data, and then we will use a function $f$, together with its parameter $\theta$, to get the predict label $\hat{Y}$, i.e. $\hat{Y}=f(\theta,X)$.

In this chapter, we use **sigmoid function**:
$$
\hat{Y}=\frac{1}{1+e^{-\theta^T{X}}}
$$
To train the parameter $\theta$, we use **gradient descent**:

![image-20210712153651510](.\imgs.\imgs\image-20210712153651510.png)

## CH2 Naive Bayes

Bayes Rule:
$$
P(Y|X)=\frac{P(X,Y)}{P(X)}=P(X|Y)\times\frac{P(Y)}{P(X)}
$$
Let's go back to the vector generation again. Now, we do not use any pre-processing techniques. For training set:

Positive:

* I am happy because I am learning NLP
* I am happy not sad

Negative:

* I am sad I am not learning NLP
* I am sad not happy

We get the frequency as follows:

![image-20210712160855685](.\imgs.\imgs\image-20210712160855685.png)

We find that in Pos, we have 13 classes while in Neg, there's only 12. Then, we compute the possibility $P(w_i|class)$ , for example, $P(I|Pos)=\frac{3}{13}=0.24$ and get another table.

In this table, some words have nearly identical conditional probability like "**I**", "**am**" and "**learning**". They don't add anything to the sentiment.

In contrast, words like "**happy**", "**sad**" and "**not**" have significant difference between probabilities. They are **power words** which express one sentiment or the other. 

Additionally, words like "**because**" only appears in one sentiment, which could not be computed in sentiment analysis. To solve this, we need to **smooth our probability function**.

Say we get a new tweet "I am happy today; I am learning." Now we use the probability table above to predict the sentiment of the tweet with the formula:
$$
\prod_{i=1}^{m}\frac{P(w_i|pos)}{P(w_i|neg)}=\frac{0.24}{0.25}\times\frac{0.24}{0.25}\times\frac{0.15}{0.08}\times\frac{0.24}{0.25}\times\frac{0.24}{0.25}\times\frac{0.08}{0.08}=1.59>1
$$
 Since the result is greater than 1, the sentiment is **positive**.