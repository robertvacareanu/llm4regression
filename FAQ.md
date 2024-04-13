# Frequently Asked Questions (FAQ)

- [Frequently Asked Questions (FAQ)](#frequently-asked-questions-faq)
  - [Should I drop method `<X>` and use LLMs for regression?](#should-i-drop-method-x-and-use-llms-for-regression)
  - [Why didn't you cite `<X>`?](#why-didnt-you-cite-x)
  - [Why didn't you try method `<X>`?](#why-didnt-you-try-method-x)
  - [Can you try other datasets?](#can-you-try-other-datasets)
  - [How to add a new dataset?](#how-to-add-a-new-dataset)
  - [Do you explain why this happens?](#do-you-explain-why-this-happens)
  - [Hasn't previous research already demonstrated that Transformers can do regression?](#hasnt-previous-research-already-demonstrated-that-transformers-can-do-regression)
  - [What if examples of Friedman #1, #2, #3 are already on the Web?](#what-if-examples-of-friedman-1-2-3-are-already-on-the-web)
  - [Why does Linear Regression perform so well in Figure 1, for example, outperforming Gradient Boosting?](#why-does-linear-regression-perform-so-well-in-figure-1-for-example-outperforming-gradient-boosting)
  - [How can I try it?](#how-can-i-try-it)



## Should I drop method `<X>` and use LLMs for regression?

No, the message of the paper is not that LLMs are better than all traditional supervised method (e.g., Gradient Boosting) and that they should be used from now on. Instead, it highlights the surprisingly powerful in-context learning capabilities of pre-trained LLMs like GPT-4 and Claude 3. That is, despite no parameter update, LLMs can (sometimes) outperform methods like the ones aforementioned, at least in small dataset regime (We tested with at most $500$ examples, as per Appendix O).

## Why didn't you cite `<X>`?

Apologize for any oversight. Missed citations will be added in the next revision.


## Why didn't you try method `<X>`?

I am open to try more methods. I mostly tried some of the most popular ones available on sklearn.

## Can you try other datasets?

Sure. 

## How to add a new dataset?

Please refer to [how_to_add_dataset.md](./how_to_add_dataset.md) on how to add a new dataset.

## Do you explain why this happens?

No, although we hypothesized that very capable LLMs emerge as very good online meta-learners. We leave the exploration of how exactly this happens to future work.

## Hasn't previous research already demonstrated that Transformers can do regression?

It has, but with one caveat: they specifically train a transformer to do these tasks. Here, on the other hand, we do not train any model. We explore the performance of fixed, already trained models like GPT-4 or Claude 3 on random regression datasets.
There has been an exploration on Iris Dataset and 2d binary classification from this [LessWrong Post](https://www.lesswrong.com/posts/c2RzFadrxkzyRAFXa/who-models-the-models-that-model-models-an-exploration-of) which I only recently become aware of and I plan to reference in the paper.

## What if examples of Friedman #1, #2, #3 are already on the Web?

We considered this and discuss it in Appendix N. One approach we used was to write our own random regression functions. This way, it is very unlikely for the models to have seen similar examples in their training data. One such example is `Original #1`, defined as: $x + 10sin(\frac{5\pi x}{100}) + 10cos(\frac{6\pi x}{100})$. We defined a total of 5 new functions. 

## Why does Linear Regression perform so well in Figure 1, for example, outperforming Gradient Boosting?

Because that dataset is linear. On non-linear datasets (e.g., see heatmap on this repository), Linear Regression performs worse.

## How can I try it?

The simplest way would be to copy-paste some of the examples from `data/prompts`. I also provide links to chats in [data/prompts/README.md](./data/prompts/README.md). Note, however, that I used the API, not the chat for the experiments. I included links to chat conversation because it is easy to share and to view. 