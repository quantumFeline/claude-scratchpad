DNN 3: Transformer GPT-OSB-20B
===

In this lab, we implement the GPT-OSS Transformer toy model and apply it to a toy example.

**Implementation**

The implementation follows instructions, with some clarifications:

* In RoPE, we can use different possible pairings: one is `0-1, 2-3, ...`, another is `0-n//2, 1-(n//2+1), 2-(n//2+2), ...`. The latter is however easier to implemented and is the one that matches the seeded test results.

* An efficient RoPE implementation requires pre-computing the trigonometric values - as they are used multiple times, this can lead to 2-3x speed of the pass.

* RoPE use is hidden inside the GroupedQueryAttention and SWAttention classes, and is not declared explicitly in the Transformer block. RoPE is applied before the key weights in order to fully enable its benefits (TODO: clarify).

* When `num_kv_heads` is unspecified, we assume it to match `num_heads`.

* MoE is optimized by calling a forward pass through the router once, then iterating over experts and applying a mask to filter the outputs relevant to the specific expert during calculation; this avoids a double or triple loop we would have needed if we also iterated over batches.

* There is some code reduplication; it is unfortunately unavoidable without going out of scope for the TODOs.

**Model training and evaluation**

We used the pre-defined number of epoch and learning rate. 

While the text prediction evaluation task is not straightforward in most real environments due to the variety of possible valid response, we are dealing with a toy example of a vocabulary consisting on only 7 words - therefore, we cann apply cross-entropy directly (after reshaping to account for batching). 

Under these conditions, the model reaches 82% accuracy.

![Loss and accuracy](./loss_acc.png)

The per-position accuracy is the following:

![Accuracy per token position](./per_pos_acc.png)

We see that the model behaviour the model has too little context to work with - it needs at least 8-10 tokens in order to achieve the baseline accuracy. The accuracy on longer text is actually higher than the overall accuracy rate would indicate, easily reaching 85-90%.

**Token generation**

In order to generate new tokens, we call the model repeatedly. Each time we provide the full sequence of the previous tokens, i.e. the model behaviour on each new tokens depends on the previous ones it has provided.

When we always pick up the most probably predicted token, the model always repeats the input token endlessly. In order to avoid that, we need to introduce nucleus sampling.

We follow the standard nucleus sampling algorithm, ensuring that the last token that makes the cumulative probability exceed `top_p` parameter is still included. 

The nucleus sampling algorithm depends on temperature. We would like to measure it numerically. For that, we go over a range of possible temperatures and estimate the average entropy of the sequence for each temperature value. To minimise noise, we do 100 generations per each temperature.

The results are the following:

![Entropy depending on temperature](./temperature_fixed.png)

We see that we need `t=0.7` or higher in order to escape the looping behavior. At `t=1`, we essentially choose the next token from the raw model logits without clipping it.

We can try raising the temperature even higher:

![Entropy depending on temperature, wide range](./temperature_wide_range.png)

An entropy of `~1.95 = log(7)` indicates that the model behaviour at that point is essentially random, so at `t >= 4` the model behaviour degrades to random.

Overall, the result suggest a viable, if a small-scale architecture that shows promising results if scaled according to the vocabulary it operates with.