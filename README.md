## Knowledge Distillation, Structural Sparsity, and Robustness of Qwen LLM

### Overview of Code
* Replaces dense modules with structurally sparse grouped linear and LoRA modules on models downloaded from Huggingface.
* Trains the weights of each individual new grouped Linear and LoRA layers on the output of the original Linear module.
* Fine-tunes the new sparse model with Knowledge Distillation from the original model using the [Platypus](https://platypus-llm.github.io/) dataset.
* Evaluates the robustness of models using the [RUPBench](https://github.com/EternityYW/RUPBench) dataset .

### Installation
Install the prerequisites:
```bash
pip install transformers
pip install datasets
pip install 'accelerate>=0.26.0'
pip install --upgrade jinja2

# For quantization of models
pip install optimum-quanto
python -m pip install --upgrade-strategy eager optimum
pip install auto-gptq
```
To run the program
```bash
python main.py
```

### Robustness of Quantization results
Robustness by model size can be seen in the table from the paper
[RUPBench: Benchmarking Reasoning Under Perturbations for Robustness Evaluation in Large Language Models]( https://arxiv.org/pdf/2406.11020
).
![RUPBench Table 2](images/RUPBench_table_%202.png "RUPBench Table 2")

The results of applying small Qwen models to the RUPBench dataset show quantization has little effect on the adversarial accuracy on going from BF16 to int8. 
Note 20% corresponds to random chance.

| Model     | Run  | Accuracy | Adversarial Accuracy | Difference |
|-----------|------|----------|----------------------|------------|
| Qwen 1.5B | BF16 | 44.37%   | 36.41%              | -7.96%     |
| Qwen 1.5B | Int8 | 43.98%   | 36.08%              | -7.90%     |
| Qwen 0.5B | BF16 | 24.47%   | 21.61%              | -2.86%     |
| Qwen 0.5B | Int8 | 26.05%   | 21.95%              | -4.10%     |

### Sparse model training results
The grouped and LoRA linear layers reduced the number of parameters of the Qwen 0.5B model from 494,032,768 to 
The teacher model correctly answers the sample question. After training the output of each sparse linear projection against the output of the original dense linear projection for 4 epochs, and fine-fine tuning on the Platypus dataset for 1 epoch, the student model is unable to answer the question.

#### Sample Question
A board game spinner is divided into three parts labeled $A$, $B$  and $C$. The probability of the spinner landing on $A$ is $\frac{1}{3}$ and the probability of the spinner landing on $B$ is $\frac{5}{12}$.  What is the probability of the spinner landing on $C$? Express your answer as a common fraction.

#### Student answer
To solve this problem, we need to find the probability of the spinner landing on any given part of the game (either $A$, $B$, or $C$). We can do this by summing up the probabilities for each part:

$$
P(A) + P(B) + P(C) = \frac{1}{3} + \frac{5}{12} + \frac{1}{6} = \frac{1}{6}
$$

Therefore, the probability of the spinner landing on any given part of the game is ${\frac{1}{6}}$. 

This solution uses the concept of probability distribution to compute the total probability of all possible outcomes. The formula for calculating the total probability of a set of events is:

$$
P(X) = \sum_{i=1}^n P(X|X) \cdot P(X)
$$

where $P(X|$ represents the probability of event X occurring in the sample, and $X$ is the set of events. In our case, we have $X = \{A\}, B\}, C. The formula becomes:

$$
P(X) = \sum_{i=1}^3 \frac{P(X|X)}{6} \cdot P(X)
$$

We can simplify this expression by dividing both sides by 6:

$$
P(X) = \frac{P(X|X)}{6} \cdot \frac{P(X)}{6} = \frac{P(X|X)}{6} \cdot \frac{P(X)}{6} = \frac{P(X|X)}{6} \cdot \frac{1}{6} = \frac{1}{36}
$$

Thus, the answer is ${\frac{1}{6}}$. This calculation is valid because the probability of the spinner landing on any given part of the game is equal to the sum of the probabilities of the three parts, which is 1/6. However, it's not correct because the probability of the spinner landing on one of the three parts is different from the other two. Therefore, the answer must be 1/6. 

In summary, the probability of the spinner landing on any given part of the game is 1/6. **Note:** The answer is valid because the probability of the spinner landing on any given part of the game is equal to the sum of the probabilities of the three parts