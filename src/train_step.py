import torch
from flexiworld.runtime import predict_adjacent_latents

K_HI = 10
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize_pixels(pixels: torch.Tensor) -> torch.Tensor:
    pixels = pixels.float().div(255.0)
    mean = pixels.new_tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1)
    std = pixels.new_tensor(IMAGENET_STD).view(1, 1, 3, 1, 1)
    return (pixels - mean) / std


def action_embeddings(model, batch):
    act_emb = model.action_encoder(batch["a_pad"], k=batch["k"])
    first = model.action_encoder(batch["prev_pad"].unsqueeze(1), k=batch["k_prev"].unsqueeze(1))
    previous = torch.cat([first, act_emb[:, :-1]], dim=1)
    return act_emb, previous


def actor_nll(
    model,
    z,
    intent,
    previous,
    batch,
    *,
    student_p: float,
    steps_remaining: torch.Tensor | None = None,
):
    batch_size, transitions = batch["k"].shape
    action_dim = int(model.primitive_dim)
    actions = batch["a_pad"].reshape(batch_size, transitions, K_HI, action_dim)

    def flatten(value):
        return value.reshape(batch_size * transitions, *value.shape[2:])

    flat_steps = None if steps_remaining is None else steps_remaining.reshape(-1)
    return model.intent_actor.nll(
        flatten(z),
        flatten(intent),
        flatten(previous),
        flatten(actions),
        flatten(batch["k"]),
        student_p=student_p,
        steps_remaining=flat_steps,
    )


def forward(self, batch, stage, cfg):
    pixels = normalize_pixels(batch["pixels"])
    batch_size, frames = pixels.shape[:2]
    encoded = self.model.encoder(
        pixels.reshape(batch_size * frames, *pixels.shape[2:]),
        interpolate_pos_encoding=True,
    )
    z = self.model.projector(encoded.last_hidden_state[:, 0]).reshape(batch_size, frames, -1)
    actions, previous = action_embeddings(self.model, batch)
    predictions = predict_adjacent_latents(self.model, z, actions, cfg.history_size)
    prediction_loss = (predictions - z[:, 1:]).square().mean()
    regularization = self.sigreg(z.transpose(0, 1))
    student_p = 0.5 if stage in {"fit", "train"} else 0.0
    local = actor_nll(
        self.model, z[:, :-1], z[:, 1:] - z[:, :-1], previous, batch, student_p=student_p
    )
    goal = actor_nll(
        self.model, z[:, :-1], z[:, -1:].detach() - z[:, :-1], previous, batch, student_p=student_p
    )
    loss = prediction_loss + 0.02 * regularization + 0.1 * local["loss"] + 0.05 * goal["loss"]
    for key, value in {
        "loss": loss,
        "pred_loss": prediction_loss,
        "sigreg_loss": regularization,
        "local_nll": local["loss"],
        "goal_nll": goal["loss"],
    }.items():
        self.log(
            f"{stage}/{key}",
            value,
            on_step=stage in {"fit", "train"},
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
    return {"loss": loss}
