# AmazonQA

**Source**: https://arxiv.org/pdf/1908.04364v1.pdf

**Problem**: Review-based QA

**Background**:

* Most questions have a response time of several days, with an average of around 2 days per question
* The product reviews are comparatively elaborate and informative as judged against
  the answers posted to specific questions
* More than half of the questions can (at least partially) be answered using existing reviews.

**Dataset**:

* Introduce a new dataset and propose a method that **combines information retrieval techniques** for **selecting relevant reviews** (given a question) and “reading comprehension” models for **synthesizing an answer** (given a question and review).
* Additional annotations, marking each question as either answerable or unanswerable based on the available reviews.
* It is extracted entirely from existing, real world data;
* It may be the largest public QA dataset with descriptive answers, consisting  of 923k questions, 3.6M answers and 14M reviews across 156k products.
* Span 17 categories of products including Electronics, Video Games, Home and Kitchen, etc.

**Processing**:

* Identify and remove chunks of duplicate text
* Remove the outliers from the dataset which are much longer than average.
* Provide query-relevant review-snippets
  * Tokenize the reviews
  * Chunk the review text into snippets
  * Rank the snippets based on the TF-IDF metric.
* Annotation
  * Answerability: whether the answer to the question is at least partially contained
    in reviews.
  * Question Type: descriptive (open-ended) or yes/no (binary).
* We make a random 80-10-10 (train-development-test) split. Each split uniformly consists of 85% of descriptive and 15% yes/no question types.

Answerability:

* A single MTurk hit is a web page shown to the workers, consisting of N = 5 questions, out of which one of them is a ‘decoy’ question.
* The responses to the decoy questions allow us to compute a lower bound of a worker’s performance.

![image-20210808134215194](.\imgs\image-20210808134215194.png)

* Loss function

$$
\frac{1}{2}\sum\abs{\text{worker\_label}-\text{true\_label}}
$$

