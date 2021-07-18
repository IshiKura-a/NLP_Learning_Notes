# S2 Probabilistic Model

## CH1 Auto-Correct

Objective: correct the misspelled words while maintaining minimum edit distance.

### How it works

1. Identify the misspelled word
2. Find strings n edit distance away
3. Filter candidates
4. Calculate word probabilities

### Identify the misspelled word

```python
if word not in vocab:
    misspelled = True
```

### Find strings n edit distance away

* Edit: an operation performed on a string to change it.
  * **insert**: add a letter
  * **delete**: remove a letter
  * **switch**: swap 2 adjacent letters
  * **replace**: change 1 letter to another

Here, we find available strings rather than words.

### Filter candidates

Here, we filter the strings that are not words.

![image-20210715202248833](.\imgs\image-20210715202248833.png)

### Calculate word probabilities

![image-20210715202930547](.\imgs\image-20210715202930547.png)

### Minimum Distance

For each operations, the edit cost varies, for example insert cost is only 1, while replace cost is 2. To transform from "play" to "stay", the cost is 4.

![image-20210715205619051](.\imgs\image-20210715205619051.png)

To compute the minimum edit distance in a formal way, we need a minimum edit distance matrix $D$ for the two words.

For example, we edit "play" to "stay". Let's denote the start of the word with '#',  The cell $D[i,j]$ means the minimum edit distance from string $\text{source}[:i]$, to string $\text{target}[:j]$.

![image-20210715211441727](D:\Mydata\Science\NLP\WED_Notes\imgs\image-20210715211441727.png)

Obviously, the cell can be computed by **dynamic programming**.

Let's start from the easiest case. For $D[i,j]$ and $D[i-1,j]$, we only need to delete the last word, that is
$$
D[i,j]=D[i-1,j]+del\_cost
$$
For $D[i,j]$ and $D[i,j-1]$, we only need to insert the last word:
$$
D[i,j]=D[i,j-1]+ins\_cost
$$
For $D[i,j]$ and $D[i-1,j-1]$, we only need to replace the last word:
$$
D[i,j]=D[i-1,j-1]+\left\{
\begin{aligned}
&rep\_cost&,\text{src}[i]\ne\text{tar}[i] \\
&0&,\text{src}[i]=\text{tar}[i] \\
\end{aligned}
\right.
$$
Mind that we need the minimum cost, hence we get the state transition equation:
$$
D[i,j]=\left\{
\begin{aligned}
&D[i-1,j]+del\_cost \\
&D[i,j-1]+ins\_cost \\
&D[i-1,j-1]+\left\{
\begin{aligned}
&rep\_cost&,\text{src}[i]\ne\text{tar}[i] \\
&0&,\text{src}[i]=\text{tar}[i] \\
\end{aligned}
\right.
\end{aligned}
\right.
$$
The result is

![image-20210715213823920](.\imgs\image-20210715213823920.png)

## CH2 Part of Speech Tagging

Part of speech tagging is to mark the words with different tags according to their property.

![image-20210716141632317](.\imgs\image-20210716141632317.png)

### Markov Chain

What is the next word after learn ?

![image-20210716142335843](D:\Mydata\Science\NLP\WED_Notes\imgs\image-20210716142335843.png)

Certainly, it's more likely to have a noun after "learn" than a verb. We can represent the likelihood by the graph below(called **Markov Chain**).

![image-20210716142636985](.\imgs\image-20210716142636985.png)

The most important part of **Markov Chain** is the state $Q=\{q_1,q_2,\cdots,q_n\}$. In the graph above, $Q=\{verb,noun\}$.

Another part of the **Markov Chain** is the **transition probabilities**, i.e. the arrow and the number on it. Transition probabilities implies the probability to transform from one state to another state. We can use a transition matrix to record the transition probabilities:

![image-20210716153146604](.\imgs\image-20210716153146604.png)

### Hidden Markov Model

Hidden Markov Model (HMM) introduce states that are hidden, or cannot directly observed. For example, the graph above has states like NN and VB. These states are hidden states for they cannot be observed by the data.

![image-20210716154540840](.\imgs\image-20210716154540840.png)

Generally, the words "jump", "run" that can be observed by the context is observable, while the tags like "verb" which cannot be observed are hidden states. We add a initial state $\pi$,

![image-20210716154801377](.\imgs\image-20210716154801377.png)

Hence, the transition probabilities can be represented by matrix $A$ which is $N+1$ by $N$. In HMM, we have an additional probability called **Emission Probability**, which is the probability to transform from a hidden state to an observable.

![image-20210716155044229](.\imgs\image-20210716155044229.png)

Accordingly, there's a matrix $B$ which is $N$ by $V$ to represent it because a word like "back" may have multiple properties.

How to calculate the probabilities?

1. Count occurrences of tag pairs

$$
C(t_{i-1},t_i)
$$

2. Calculate probabilities using counts

$$
P(t_i|t_{i-1})=\frac{C(t_{i-1},{t_i})}{\sum_{j=1}^NC(t_{i-1},t_j)}
$$

​	In case all the counts are zero, we add an $\epsilon$ to each count for **smoothing**, that's
$$
P(t_i|t_{i-1})=\frac{C(t_{i-1},{t_i})+\epsilon}{\sum_{j=1}^NC(t_{i-1},t_j)+N\epsilon}
$$

### Viterbi Algorithm

![image-20210716160922144](.\imgs\image-20210716160922144.png)

To pass a sentence "I love to learn", we may have such a sequence $\pi\rightarrow{O}\rightarrow{VB}\rightarrow{O}\rightarrow{VB}$ with probability 0.0003. The Viterbi Algorithm actually computes several such passes at the same time to find the most likely sequence. It can be expressed into 3 steps:

1. Initialization step.
2. Forward pass.
3. Backward pass.

During the steps, we need two auxiliary matrices $C$ and $D$. 

#### Step 1 

In the first column of $C$, we calculate the probability from $\pi$ to word $w_1$, that is
$$
c_{i,1}=\pi_i\times{b_{i,cindex(w_1)}}=a_{1,i}\times{b_{i,cindex(w_1)}}
$$
In the $D$ matrix, we store the labels that represent the different state we are traversing when finding the most likely sequence of POS tags for the given words $w_1$ to $w_k$. In the first column, we simply set $d_{i,1}=0$ as there're no preceding POS tags we have traversed.

 #### Step 2

In the forward pass
$$
\begin{aligned}
c_{i,j}&=\max_{k}c_{k,j-1}\times{a}_{k,i}\times{b}_{i,cindex{(w_j)}} \\
d_{i,j}&=\arg\max_{k}c_{k,j-1}\times{a}_{k,i}\times{b}_{i,cindex{(w_j)}} \\
\end{aligned}
$$

#### Step 3

Using the matrix $C$, we can get
$$
s=\arg\max_{i}c_{i,K}
$$
For example, $s=1$, and for the matrix $D$ below, we can restore the sequence:

![image-20210716163400544](.\imgs\image-20210716163400544.png)

The algorithm stops when we find the start token.

> In case the probabilities are very small, use **log probabilities** instead.

## CH3 N-gram

An N-gram is a sequence of N words.

For example, a corpus is "I am happy because I am learning".

Unigrams are {I, am, happy, because, learning}.

Bigrams are {I am, am happy, happy because, ...}.

Trigrams are {I am happy, am happy because, ...}.

Let's denote the $n^{th}$ word to be $w_n$; $w_i^n=w_iw_{i+1}\cdots{w_{i+n-1}}$.

The probability of unigram is
$$
P(w)=\frac{C(w)}{m}
$$
The probability of bigram is
$$
P(y|x)=\frac{C(xy)}{\sum_{w}{C(xw)}}=\frac{C(xy)}{C(x)}
$$
The probability of trigram is
$$
P(w_3|w_1^2)=\frac{C(w_1^2w_3)=C(w_1w_2w_3)=C(w_1^3)}{C(w_1^2)}
$$
All in all, the probability of N-gram is
$$
P(w_N|w_1^{N-1})=\frac{C(w_1^{N-1}w_N)=C(w_1^N)}{C(w_1^{N-1})}
$$
Using **Chain Rule**:
$$
P(A,B,C,D)=P(A)P(B|A)P(C|A,B)P(D|A,B,C)
$$
We can calculate the probability of a certain sequence. The problem is that corpus almost never contains the exact sentence we're interested in or even its longer subsequences. For example, 
$$
P(\text{tea}|\text{the teacher drinks})=\frac{C(\text{the teacher drinks tea})}{C(\text{the teacher drinks})}
$$
Both of the numerator and denominator are almost zero.

For approximation, we have
$$
P(\text{tea}|\text{the teacher drinks})\approx{P(\text{tea}|\text{drinks})}
$$
This is called Markov assumption.

Thus,
$$
P(w_1^n)\approx{\prod_{i=1}^n}{P(w_i|w_{i=1})}
$$
Accordingly,
$$
P(\text{the teacher drinks tea})\approx{P(\text{the})}P(\text{teacher}|\text{the})P(\text{drinks}|\text{teacher})P(\text{tea}|\text{drinks})
$$
For we do not have the context of "the", we can only give the prediction of it. As a result, we add a start symbol $<s>$ to the corpus, so that we can calculate the probability of bigram.
$$
P(\text{<s>the teacher drinks tea})\approx{P(\text{the}|\text{<s>})}P(\text{teacher}|\text{the})P(\text{drinks}|\text{teacher})P(\text{tea}|\text{drinks})
$$
For N-gram model, we need to add N-1 start tokens $$<s>$$.

In case we calculate the probability that at the end of the corpus, the probability would be 1. To avoid this, we add a end symbol $</s>$.

For bigram:

![image-20210717162102002](.\imgs\image-20210717162102002.png)

For N-gram, only one $</s>$ is needed.

![image-20210717162703397](.\imgs\image-20210717162703397.png)

To build a language model, we need count matrix and probability matrix.

![image-20210717163426020](.\imgs\image-20210717163426020.png)

![image-20210717163447751](.\imgs\image-20210717163447751.png)

From the probability matrix, we can get sentence probability. For example, 

![image-20210717163658578](.\imgs\image-20210717163658578.png)

However, the product may be zero, which will lead to underflow. To avoid underflow, we need log probability.

Generally, the algorithm is:

1. Choose sentence start
2. Choose next bigram starting with the previous word
3. Continue until $</s>$ is picked.

For model training, we need to split corpus to **Train/Validation/Test**. For smaller corpora, we divide into:

* 80% Train
* 10% Validation
* 10% Test

For larger corpora, we divide into:

* 98% Train
* 1% Validation
* 1% Test

We can use **perplexity** to evaluate the test set,

![image-20210717170244669](.\imgs\image-20210717170244669.png)

The higher the language model estimates the probability of test set, the lower the perplexity is going to be.

In a bigram model, the perplexity is
$$
PP(W)=\sqrt[m]{\prod_{i=1}^m\prod_{j=1}^{|s_i|}\frac{1}{P(w_j^{(i)}|w_{j-1}^{(i)})}}
$$
where $w_{j}^{(i)}$ is the $j^{th}$ word in the $i^{th}$ sentence. For simplicity, we concatenate all sentences in W
$$
PP(W)=\sqrt[m]{\prod_{i=1}^m\frac{1}{P(w_i|w_{i-1})}}
$$
Also, we have log perplexity:
$$
\log{PP(W)}=-\frac{1}{m}\sum_{i=1}^m\log(P(w_i|w_{i-1}))
$$
