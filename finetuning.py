import torch
from torch import nn
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from sae_lens import SAE
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

# ================= 1. Load the instruct model and tokenizer =================
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct").to(device)
for param in model.parameters():
    param.requires_grad = False  # Freeze instruct model weights

# ================= 2. Load the pretrained SAE =================
release = "llama-3-8b-it-res-jh"
sae_id = "blocks.25.hook_resid_post"
pretrained_sae, cfg_dict, sparsity = SAE.from_pretrained(release, sae_id)
print("Pretrained SAE config:")
print(f"d_in: {cfg_dict['d_in']}, d_sae: {cfg_dict['d_sae']}, explanation_factor: {cfg_dict.get('explanation_factor')}")

# ================= 3. Create a composite model that uses the instruct model to generate activations, =================
#    then fine-tunes the SAE on those activations via a reconstruction objective.
class InstructSAEModel(nn.Module):
    def __init__(self, instruct_model: nn.Module, sae: nn.Module, hook_index: int):
        """
        :param instruct_model: The instruct Llama model (frozen).
        :param sae: The SAE module to fine-tune.
        :param hook_index: Index of the block from which to capture activations.
        """
        super().__init__()
        self.instruct_model = instruct_model
        self.sae = sae
        self.hook_index = hook_index  # e.g. if SAE was originally on "blocks.25", use index 24 (0-indexed)

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Define a hook to capture the activation from the target block.
        activation = {}

        def capture_hook(module, inputs, output):
            activation["value"] = output
            return output

        # Register the hook on the desired block.
        handle = self.instruct_model.model.blocks[self.hook_index].register_forward_hook(capture_hook)

        # Forward pass through instruct model.
        outputs = self.instruct_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        handle.remove()  # Remove hook immediately after capture

        if "value" not in activation:
            raise RuntimeError("Failed to capture activation from instruct model block.")

        # Feed the captured activation through the SAE.
        # (Assuming SAE returns a reconstruction of its input.)
        recon = self.sae(activation["value"])

        # Compute reconstruction loss (mean squared error).
        loss = ((recon - activation["value"]) ** 2).mean()
        return {"loss": loss, "logits": outputs.logits}  # logits are passed along for logging if needed

# Use block index 24 assuming 0-indexing for "blocks.25.hook_resid_post"
hook_index = 24
composite_model = InstructSAEModel(model, pretrained_sae, hook_index)

# 4. ================= Prepare your dataset. =================
# For demonstration, we use a small slice of wikitext-2.
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# 5. ================= Set up training arguments. =================
training_args = TrainingArguments(
    output_dir="./finetuned_instruct_sae",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=50,
    learning_rate=2e-5,
    weight_decay=0.01,
)

# 6. ================= Create an optimizer that only updates SAE parameters. =================
optimizer = torch.optim.AdamW(composite_model.sae.parameters(), lr=training_args.learning_rate)

# 7. ================= Set up the Trainer. =================
trainer = Trainer(
    model=composite_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    optimizers=(optimizer, None),  # Scheduler is left as None for simplicity.
)

print("Starting SAE fine-tuning on instruct model... (This might be the most caffeinated moment of your day!)")
trainer.train()
print("Fine-tuning complete. Saving instruct SAE weights...")

# 8. ================= Save the fine-tuned SAE. =================
torch.save(pretrained_sae.state_dict(), "./instruct_sae.pt")
print("Instruct SAE saved to './instruct_sae.pt'")