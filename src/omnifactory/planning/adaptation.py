"""Compile component policies against an explicit checkpoint parameter inventory.

This produces a reviewable plan. It does not inject adapters or quantize weights.
"""

from math import prod
from typing import Annotated, Literal

from pydantic import Field, model_validator

from omnifactory.data.schema import Identifier, Record
from omnifactory.errors import OmniFactoryError
from omnifactory.models.registry import ModelSpec


class ParameterSpec(Record):
    name: Identifier
    shape: Annotated[tuple[Annotated[int, Field(gt=0)], ...], Field(min_length=1)]
    kind: Literal["linear_weight", "parameter"] = "parameter"
    storage_id: str | None = None
    quantized: bool = False

    @property
    def numel(self) -> int:
        return prod(self.shape)


class ComponentPolicy(Record):
    mode: Literal["frozen", "lora", "full"] = "frozen"
    lr: Annotated[float, Field(gt=0, allow_inf_nan=False)] | None = None
    rank: Annotated[int, Field(gt=0)] | None = None
    alpha: Annotated[float, Field(gt=0, allow_inf_nan=False)] | None = None
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")

    @model_validator(mode="after")
    def policy_fields(self):
        if self.mode == "frozen" and any(v is not None for v in (self.lr, self.rank, self.alpha)):
            raise ValueError("frozen components cannot specify lr, rank or alpha")
        if self.mode != "frozen" and self.lr is None:
            raise ValueError("trainable components require an explicit lr")
        if self.mode == "lora" and (
            self.rank is None or self.alpha is None or not self.target_modules
        ):
            raise ValueError("LoRA requires rank, alpha and target_modules")
        if self.mode == "full" and (self.rank is not None or self.alpha is not None):
            raise ValueError("full training cannot specify LoRA rank/alpha")
        return self


class AdaptationRequest(Record):
    model: Identifier
    components: dict[str, ComponentPolicy]
    loss_components: Annotated[tuple[str, ...], Field(min_length=1)] = ("language_model",)


class ParameterGroup(Record):
    component: str
    mode: Literal["lora", "full"]
    lr: float
    parameters: tuple[str, ...]
    base_numel: int
    trainable_numel: int
    rank: int | None
    alpha: float | None


class AdaptationPlan(Record):
    model: str
    contract_version: str
    status: Literal["inventory_plan_only"] = "inventory_plan_only"
    groups: tuple[ParameterGroup, ...]
    trainable_numel: int
    gradient_components: tuple[str, ...]
    cacheable_components: tuple[str, ...]
    unassigned_parameters: tuple[str, ...]
    warnings: tuple[str, ...]


def _owner(name: str, spec: ModelSpec) -> str | None:
    matches = [
        (len(prefix), component.name)
        for component in spec.components
        for prefix in component.prefixes
        if name == prefix or name.startswith(prefix + ".")
    ]
    if not matches:
        return None
    longest = max(length for length, _ in matches)
    owners = {component for length, component in matches if length == longest}
    if len(owners) != 1:
        raise OmniFactoryError("E_COMPONENT_ALIAS", f"ambiguous ownership for {name}")
    return owners.pop()


def compile_adaptation(
    spec: ModelSpec,
    request: AdaptationRequest,
    inventory: list[ParameterSpec],
) -> AdaptationPlan:
    if request.model not in (spec.key, spec.model_id):
        raise OmniFactoryError("E_MODEL", "adaptation request and model contract differ")
    names = {c.name for c in spec.components}
    unknown = (set(request.components) | set(request.loss_components)) - names
    if unknown:
        raise OmniFactoryError("E_COMPONENT", f"unknown components: {sorted(unknown)}")
    if len({p.name for p in inventory}) != len(inventory):
        raise OmniFactoryError("E_INVENTORY", "duplicate parameter names")
    policies = {name: request.components.get(name, ComponentPolicy()) for name in names}
    trainable = {name for name, p in policies.items() if p.mode != "frozen"}
    if not trainable:
        raise OmniFactoryError("E_EMPTY_TRAINING", "select at least one trainable component")
    graph = {c.name: c.inputs for c in spec.components}

    def ancestors(name: str) -> set[str]:
        # Interleaved attention/expert/router modules are one conservative
        # gradient dependency unit. Visit peers as well as DAG parents; the
        # visited set handles the cycles introduced by those peer relations.
        result, pending = set(), [name]
        while pending:
            current = pending.pop()
            if current in result:
                continue
            result.add(current)
            pending.extend(graph[current])
            for group in spec.gradient_couplings:
                if current in group:
                    pending.extend(group)
        return result

    used = set().union(*(ancestors(name) for name in request.loss_components))
    if trainable - used:
        raise OmniFactoryError(
            "E_UNUSED_COMPONENT", f"not on a loss path: {sorted(trainable - used)}"
        )
    gradient = {name for name in used if ancestors(name) & trainable}
    cacheable = used - gradient
    owned: dict[str, list[ParameterSpec]] = {name: [] for name in names}
    unassigned = []
    aliases: dict[str, list[tuple[ParameterSpec, str | None]]] = {}
    for parameter in inventory:
        owner = _owner(parameter.name, spec)
        if owner is None:
            unassigned.append(parameter.name)
        else:
            owned[owner].append(parameter)
        aliases.setdefault(parameter.storage_id or parameter.name, []).append((parameter, owner))

    # Tied trainable weights need a dedicated model mapping; never optimize aliases twice.
    for group in aliases.values():
        if len(group) > 1 and any(owner in trainable for _, owner in group):
            raise OmniFactoryError(
                "E_SHARED_TRAINABLE",
                f"explicit tied-weight handling required: {[p.name for p, _ in group]}",
            )
    groups = []
    for name in sorted(trainable):
        policy = policies[name]
        selected = owned[name]
        if policy.mode == "lora":
            selected = [
                p
                for p in selected
                if p.kind == "linear_weight"
                and len(p.shape) == 2
                and p.name.removesuffix(".weight").rsplit(".", 1)[-1] in policy.target_modules
            ]
        if not selected:
            raise OmniFactoryError(
                "E_NO_MATCH", f"{name}: no parameters matched policy in this inventory"
            )
        if policy.mode == "full" and any(p.quantized for p in selected):
            raise OmniFactoryError(
                "E_QUANTIZED_FULL", f"{name}: full training of quantized parameters is unsupported"
            )
        if policy.mode == "lora" and any(policy.rank > min(p.shape) for p in selected):
            raise OmniFactoryError(
                "E_LORA_RANK", f"{name}: rank exceeds an adapted matrix dimension"
            )
        base_numel = sum(p.numel for p in selected)
        tuned = (
            sum(policy.rank * sum(p.shape) for p in selected)
            if policy.mode == "lora"
            else base_numel
        )
        groups.append(
            ParameterGroup(
                component=name,
                mode=policy.mode,
                lr=policy.lr,
                parameters=tuple(sorted(p.name for p in selected)),
                base_numel=base_numel,
                trainable_numel=tuned,
                rank=policy.rank,
                alpha=policy.alpha,
            )
        )
    warnings = [
        "The inventory must come from the exact checkpoint; this plan does not load or train it."
    ]
    if gradient - trainable:
        warnings.append("Frozen components on the gradient path must not be wrapped in no_grad().")
    if unassigned:
        warnings.append(
            "Unassigned parameters remain frozen; inspect the reported names before execution."
        )
    return AdaptationPlan(
        model=spec.key,
        contract_version=spec.contract_version,
        groups=tuple(groups),
        trainable_numel=sum(g.trainable_numel for g in groups),
        gradient_components=tuple(sorted(gradient)),
        cacheable_components=tuple(sorted(cacheable)),
        unassigned_parameters=tuple(sorted(unassigned)),
        warnings=tuple(warnings),
    )
