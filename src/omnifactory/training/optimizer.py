"""Optional FP32 AdamW master parameters without changing model execution dtype."""

import torch

from omnifactory.errors import OmniFactoryError


class MasterAdamW(torch.optim.Optimizer):
    """Keep sub-ULP updates and Adam moments in FP32; expose original parameters.

    Gradients are clipped on the model by the trainer, then copied to masters.
    Checkpoints retain masters plus moments; projected model weights alone cannot
    resume this optimizer exactly. Group order is bound by the training contract.
    """

    def __init__(self, groups, *, weight_decay=0.0):
        super().__init__(groups, dict(lr=1e-3, weight_decay=weight_decay))
        master_groups = []
        for group in self.param_groups:
            masters = []
            for p in group["params"]:
                if p.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                    raise ValueError("FP32 masters require FP16/BF16/FP32 trainable parameters")
                masters.append(torch.nn.Parameter(p.detach().float().clone()))
            master_groups.append({**group, "params": masters})
        self.inner = torch.optim.AdamW(master_groups, weight_decay=weight_decay)

    def zero_grad(self, set_to_none=True):
        super().zero_grad(set_to_none=set_to_none)
        self.inner.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if len(self.param_groups) != len(self.inner.param_groups):
            raise ValueError("adding parameter groups after construction is unsupported")
        for group, master_group in zip(self.param_groups, self.inner.param_groups, strict=True):
            # The trainer owns per-component LR scheduling on the outer groups.
            master_group.update({k: v for k, v in group.items() if k != "params"})
            for p, master in zip(group["params"], master_group["params"], strict=True):
                master.grad = None if p.grad is None else p.grad.detach().float()
        self.inner.step()
        for group, master_group in zip(self.param_groups, self.inner.param_groups, strict=True):
            for p, master in zip(group["params"], master_group["params"], strict=True):
                if p.grad is not None:
                    p.copy_(master.to(dtype=p.dtype))
        return loss

    def state_dict(self):
        return dict(
            format="omnifactory-fp32-master-adamw-v1",
            master_weights=[
                [p.detach().clone() for p in g["params"]] for g in self.inner.param_groups
            ],
            parameter_dtypes=[[str(p.dtype) for p in g["params"]] for g in self.param_groups],
            optimizer=self.inner.state_dict(),
        )

    @torch.no_grad()
    def load_state_dict(self, state_dict):
        if state_dict.get("format") != "omnifactory-fp32-master-adamw-v1":
            raise OmniFactoryError(
                "E_OPTIMIZER_STATE",
                "FP32-master resume requires its original master weights and moments",
            )
        weights = state_dict["master_weights"]
        expected = [[str(p.dtype) for p in g["params"]] for g in self.param_groups]
        if state_dict["parameter_dtypes"] != expected or len(weights) != len(self.param_groups):
            raise OmniFactoryError("E_OPTIMIZER_STATE", "master group/dtype mismatch")
        # Validate the complete state before mutating either masters or moments.
        for group, masters, saved in zip(
            self.param_groups, self.inner.param_groups, weights, strict=True
        ):
            if len(saved) != len(group["params"]):
                raise OmniFactoryError("E_OPTIMIZER_STATE", "master parameter count mismatch")
            for p, master, value in zip(group["params"], masters["params"], saved, strict=True):
                if (
                    value.shape != master.shape
                    or value.dtype != torch.float32
                    or not torch.isfinite(value).all()
                ):
                    raise OmniFactoryError("E_OPTIMIZER_STATE", "invalid FP32 master parameter")
                if not torch.equal(value.to(device=p.device, dtype=p.dtype), p):
                    raise OmniFactoryError(
                        "E_OPTIMIZER_STATE", "master projection differs from restored model weights"
                    )
        self.inner.load_state_dict(state_dict["optimizer"])
        for group, masters, saved in zip(
            self.param_groups, self.inner.param_groups, weights, strict=True
        ):
            group.update({k: v for k, v in masters.items() if k != "params"})
            for master, value in zip(masters["params"], saved, strict=True):
                master.copy_(value)


def create_optimizer(groups, config, *, weight_decay=None):
    decay = config.weight_decay if weight_decay is None else weight_decay
    cls = MasterAdamW if config.optimizer_master_weights else torch.optim.AdamW
    return cls(groups, weight_decay=decay)
