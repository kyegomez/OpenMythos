This PR adds several high-confidence improvements to the OpenMythos architecture:

- **Flash Attention**: Upgraded GQAttention and MLAttention to use PyTorch native `F.scaled_dot_product_attention` for better memory complexity and speed.
- **Hugging Face compatibility**: Wrapped models with `PreTrainedModel` and `PretrainedConfig`.
- **Core Exports**: Added init file to the open_mythos directory for simpler imports.
- **Training loop and MoE Loss**: Added an initial training script providing load balancing logic for MoE models.

These changes provide immediate value and make the model faster and easier to use with existing AI ecosystems.