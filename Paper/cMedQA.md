# CMedQA

**Source**: https://www.mdpi.com/2076-3417/7/8/767

**Problem**: Chinese medical question answer matching

**Attribute**:

* Avoid Chinese word segmentation in text preprocessing
* Introduce CNN to extract contextual information
* Can be trained with minimal human supervision and does not require any
  handcrafted features, rule-based patterns, or external resources

**Why not word segmentation**?

* Domain-specific future causes very sharp accuracy decline to those general-purpose word segmentation tools when being directly applied to medical texts
* Inappropriate when coping with the user-composed and unedited medical questions and answers posted on online communities,

**Solution**: Character embedding

![image-20210807211015559](.\imgs\image-20210807211015559.png)

**Merit**:

* Reduce the chances of errors brought about by word segmentation algorithms
* The number of unseen or rare characters is also much smaller

![image-20210807211130241](.\imgs\image-20210807211130241.png)

**Dataset**:

* Collecting questions and answers from an online Chinese medical questions answering website (http://www.xywy.com)
* Questions posted by users mainly contain **the description of symptoms, diagnosis and treatment of diseases, use of drugs, psychological counseling**, etc.
* Only certified doctors are permitted to answer questions.
* Consists of 54, 000 questions and more than 101, 000 answers.

**Algorithm**: Multi-CNN

* In single CNN, the scale of information extracted from the model is limited.
* In multi-CNN, we concatenate output vectors from different-scale CNNs as the final representation of the question or the answer.

![image-20210807212031657](.\imgs\image-20210807212031657.png)