import torch
import lightning as pl


def configure_deterministic_math(seed: int) -> None:
    """Match the deterministic Math-SDPA contract used for paper checkpoints."""
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    pl.seed_everything(seed, workers=True)


def predict_adjacent_latents(model, embeddings, action_embeddings, history_size):
    """Predict every adjacent transition with a bounded history window."""
    transitions = embeddings.size(1) - 1
    if transitions < 1:
        raise ValueError("A training window must contain at least two frames")

    prefix = min(history_size, transitions)
    predictions = [model.predict(embeddings[:, :prefix], action_embeddings[:, :prefix])]
    for index in range(prefix, transitions):
        start = max(0, index - history_size + 1)
        prediction = model.predict(
            embeddings[:, start : index + 1],
            action_embeddings[:, start : index + 1],
        )[:, -1:]
        predictions.append(prediction)
    return torch.cat(predictions, dim=1)
