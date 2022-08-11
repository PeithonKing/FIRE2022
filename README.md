# FIRE 2022

### The problem presented to us was sentiment classification in nature

&emsp; Given the massive success of Transformers in the field of NLP, we decided to use Transformer-based architectures for our model. We used a variant of the Bert Model. To build our pipeline, we used the **Hugging face interface** since it already has state-of-the-art architectures predefined. We used **Pytorch (Vanilla)** for training a **Pre-trained Bert** model which was specifically trained on a sentiment analysis task of tweets. Since our data corpus was very small we did not try to train the architecture from scratch. We also used some **state-of-the-art data augmentation techniques** to increase our data corpus and make our training more stable. We used **AdamW optimiser** along with a **cosine decay** schedule for our training.
