# Laplace LoRA

This is the code for the course project done as part of the CS772: Probabilistic Machine Learning under the guidance of [Prof. Piyush Rai](https://www.cse.iitk.ac.in/users/piyush/).
This is an extension to the work: [Bayesian Low-Rank Adaptation for Large Language Models](https://arxiv.org/pdf/2308.13111).
The parent code repository can be found [here](https://github.com/MaximeRobeyns/bayesian_lora).

## Installation

```bash
pip install bayesian-lora
```

# Usage

To use this we need to run the script in `examples/example_usage_modified.py`. This has the default settings as discussed in the [report](https://github.com/ujwalk04/laplace-lora/blob/main/CS772_Report_.pdf). Any changes required can be done by changing the config files in the `examples/configs/`.

The script is running using LlaMA-3 on Winogrande. Also using Meta-LlaMA requires configuring with huggingface access token. Update it using:
```bash
python -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('MY_HUGGINGFACE_TOKEN_HERE')"
```
Note that running this requires a local installation with a few extra
dependencies. Run:
```bash
git clone https://github.com/MaximeRobeyns/bayesian_lora
cd bayesian_lora
pip install -e ".[examples]"
```
and then
```bash
python ./examples/example_usage_modified.py
```

The finetuning script (without laplace-lora) is present at `examples/finetune.py`. We can use it to perform normal lora fine-tuning on LLMs.
```bash
python ./examples/finetune.py
```
<hr>

#### From the original repository:

The main functions this library provides are for calculating Kronecker factors,
the marginal likelihood, and the posterior predictive distribution. We show how
to use these in the examples below.

## Calculating (low-rank) Kronecker factors

First, wrap your model call in a function that takes a batch from your data
loader, and returns the relevant logits. For a CausalLM from HuggingFace:

```python
def fwd_call(model: nn.Module, batch_prompts: Any) -> t.Tensor:
    inputs = tokenizer(batch_prompts).to(device)
    outputs = model(**inputs)
    logits = outputs.logits[:, -1]  # Get the last token logits
    return logits
```
You can now call our `calculate_kronecker_factors` function:
```python
from bayesian_lora import calculate_kronecker_factors

factors = calculate_kronecker_factors(
    model,            # Your model (not necessarily PEFT)
    fwd_call,         # Model call wrapper, defined above
    train_loader,     # Your training data loader
    cfg.n_kfac,       # (Optional) rank to use
    cfg.lr_threshold, # (Optional) threshold for low-rank approximation
    ["lora"],         # modules to target
    use_tqdm=True,    # (Optional) use tqdm for progress bar
)
```
In the above, the `["lora"]` argument contains a case-insensitive list of
keywords to identify modules to target. Since we're working with a LoRa model,
we choose `"lora"` to target (e.g. `layers.0.q_proj.lora_A`, etc).

The `factors` are a dictionary with keys being the full name of the targetted
modules, and a tuple of two tensors as the values: the first being the
(possibly low-rank) Kronecker factor corresponding to the input activations,
and the second being the (possibly low-rank) factor corresponding to the output
gradients.

See [the K-FAC docs](https://maximerobeyns.github.io/bayesian_lora/kfac.html)
for more detail.

## Model Evidence

We provide a function called `model_evidence` which returns the evidence /
marginal likelihood.

```python
from bayesian_lora import model_evidence

evidence = model_evidence(
    model,           # Your model
    log_likelihood,  # A Tensor with model's log likelihood on some eval dataset
    factors,         # Kronecker factors, as calculated above
    n_lora,          # rank used in the LoRA adapters
    n_kfac,          # rank used in the Kronecker factors
    prior_var,       # prior variance hyperparameter, as a tensor
)
```

You can then use `evidence` as the loss in a normal training loop, presuming
your parameters (e.g. `prior_var` have gradients).

## Posterior Predictive Distribution

To get the parameters of the Gaussian over the logits, use
the `jacobian_mean` and `variance` functions.

```python
with t.no_grad():
    for batch in validation_loader
        prompts, classes = batch

        batch_inputs = tokenizer(prompts)

        # Predict the output logit locations
        # target_ids is a tensor containing the indices of the target tokens
        # e.g. [354, 355, 356].
        jacobian, f_mu = jacobian_mean(
            model, batch_inputs, target_ids
        )

        # Predict the output logit variances
        f_var = variance(
            batch_inputs,     # inputs
            jacobian,         # the Jacobian dictionary, obtained above
            factors,          # Kronecker factors, as calculated above
            prior_var,        # prior variance hyperparameter, as a tensor
            classes.size(-1), # number of classes to predict
            n_lora,           # rank of the LoRA adapters
            n_kfac,           # rank of the Kronecker factors
            device,           # device to use
        )

        # Now use the parameters to e.g. sample logits from the Gaussian
        # predictive, parametrised by f_mu, f_var
        L = t.linalg.cholesky(f_var)
        samples = 100_000
        f_mu = f_mu.expand(samples, *f_mu.shape)
        L = L.expand(samples, *L.shape)
        eps = t.randn_like(f_mu)
        logits = (f_mu + L @ eps).squeeze(-1).softmax(-1).mean(0)
```

The above is a minimal example; see [this
section](https://maximerobeyns.github.io/bayesian_lora/bayesian_lora.html#posterior-predictive)
of the documentation for more detail.

# Development

This library is intentionally very small and hackable. It has two main files,
and three dependencies (`torch`, `tqdm` and `jaxtyping`.)

- `main.py` contains methods specific to [the paper](https://openreview.net/forum?id=FJiUyzOF1m),
- `kfac.py` contains relatively portable K-FAC methods

Feel free to directly copy the code into your projects and hack on it.
