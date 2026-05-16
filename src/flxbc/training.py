from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from flxbc.data import ArrayDataset
from flxbc.strategy import ParameterDict


@dataclass(slots=True)
class LocalTrainResult:
    node_id: str
    parameters: ParameterDict
    samples: int
    metrics: dict[str, float]
    update_norm: float


class TrainingBackend(Protocol):
    name: str

    def initial_parameters(self, dataset: ArrayDataset) -> ParameterDict: ...

    def train_local(
        self,
        *,
        node_id: str,
        global_parameters: ParameterDict,
        train: ArrayDataset,
        val: ArrayDataset,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        proximal_mu: float,
        strategy: str,
        seed: int,
    ) -> LocalTrainResult: ...

    def evaluate(self, parameters: ParameterDict, dataset: ArrayDataset) -> dict[str, float]: ...


def select_backend(*, requested_device: str = "auto") -> TrainingBackend:
    if requested_device == "numpy":
        return NumpyPrototypeBackend()
    try:
        import torch  # noqa: F401

        return TorchBackend(device=requested_device)
    except Exception:
        return NumpyPrototypeBackend()


class NumpyPrototypeBackend:
    name = "numpy-prototype"

    def initial_parameters(self, dataset: ArrayDataset) -> ParameterDict:
        feature_shape = dataset.images.shape[1:]
        return {
            f"prototype_{class_id}": np.zeros(feature_shape, dtype=np.float32)
            for class_id in range(dataset.num_classes)
        }

    def train_local(
        self,
        *,
        node_id: str,
        global_parameters: ParameterDict,
        train: ArrayDataset,
        val: ArrayDataset,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        proximal_mu: float,
        strategy: str,
        seed: int,
    ) -> LocalTrainResult:
        del batch_size, learning_rate
        parameters = {name: value.copy() for name, value in global_parameters.items()}
        rng = np.random.default_rng(seed)
        for class_id in range(train.num_classes):
            mask = train.labels == class_id
            if not np.any(mask):
                continue
            local_mean = train.images[mask].mean(axis=0).astype(np.float32)
            noise_std = 0.12 / max(1, np.sqrt(float(mask.sum())))
            local_mean = local_mean + rng.normal(0, noise_std, size=local_mean.shape).astype(
                np.float32
            )
            key = f"prototype_{class_id}"
            blend = 0.55 if strategy in {"fedprox", "bc-ca-fedprox"} else 0.7
            if epochs > 1:
                blend = min(0.9, blend + 0.05 * (epochs - 1))
            if proximal_mu > 0 and strategy in {"fedprox", "bc-ca-fedprox"}:
                blend = max(0.2, blend - proximal_mu)
            parameters[key] = (1.0 - blend) * parameters[key] + blend * local_mean

        metrics = self.evaluate(parameters, val)
        update_norm = _parameter_distance(parameters, global_parameters)
        return LocalTrainResult(
            node_id=node_id,
            parameters=parameters,
            samples=len(train.labels),
            metrics=metrics,
            update_norm=update_norm,
        )

    def evaluate(self, parameters: ParameterDict, dataset: ArrayDataset) -> dict[str, float]:
        predictions = []
        probabilities = []
        prototypes = [
            parameters[f"prototype_{class_id}"] for class_id in range(dataset.num_classes)
        ]
        for image in dataset.images:
            distances = [float(np.mean((image - prototype) ** 2)) for prototype in prototypes]
            probs = _softmax(-np.asarray(distances, dtype=np.float64))
            probabilities.append(probs)
            predictions.append(int(np.argmin(distances)))
        return _classification_metrics(
            np.asarray(predictions),
            dataset.labels,
            dataset.num_classes,
            probabilities=np.asarray(probabilities),
        )


class TorchBackend:
    name = "torch-cnn"

    def __init__(self, *, device: str = "auto") -> None:
        import torch

        self.torch = torch
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

    def initial_parameters(self, dataset: ArrayDataset) -> ParameterDict:
        model = self._build_model(dataset)
        return self._state_to_numpy(model.state_dict())

    def train_local(
        self,
        *,
        node_id: str,
        global_parameters: ParameterDict,
        train: ArrayDataset,
        val: ArrayDataset,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        proximal_mu: float,
        strategy: str,
        seed: int,
    ) -> LocalTrainResult:
        torch = self.torch
        torch.manual_seed(seed)
        model = self._build_model(train)
        model.load_state_dict(self._numpy_to_state(global_parameters))
        model.to(self.device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        global_tensors = {
            name: tensor.detach().clone().to(self.device)
            for name, tensor in model.state_dict().items()
            if tensor.is_floating_point()
        }

        x_train = self._tensor_images(train)
        y_train = torch.tensor(train.labels, dtype=torch.long)
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(y_train), generator=generator)
        for _ in range(epochs):
            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start : start + batch_size]
                xb = x_train[batch_idx].to(self.device)
                yb = y_train[batch_idx].to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                if strategy in {"fedprox", "bc-ca-fedprox"} and proximal_mu > 0:
                    prox = torch.zeros((), device=self.device)
                    current_state = model.state_dict()
                    for name, original in global_tensors.items():
                        prox = prox + torch.sum((current_state[name] - original) ** 2)
                    loss = loss + 0.5 * proximal_mu * prox
                loss.backward()
                optimizer.step()

        parameters = self._state_to_numpy(model.state_dict())
        metrics = self.evaluate(parameters, val)
        return LocalTrainResult(
            node_id=node_id,
            parameters=parameters,
            samples=len(train.labels),
            metrics=metrics,
            update_norm=_parameter_distance(parameters, global_parameters),
        )

    def evaluate(self, parameters: ParameterDict, dataset: ArrayDataset) -> dict[str, float]:
        torch = self.torch
        model = self._build_model(dataset)
        model.load_state_dict(self._numpy_to_state(parameters))
        model.to(self.device)
        model.eval()
        x = self._tensor_images(dataset).to(self.device)
        y = torch.tensor(dataset.labels, dtype=torch.long).to(self.device)
        with torch.no_grad():
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y).item() if len(y) else 0.0
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        metrics = _classification_metrics(
            preds,
            dataset.labels,
            dataset.num_classes,
            probabilities=probabilities,
        )
        metrics["loss"] = float(loss)
        return metrics

    def _build_model(self, dataset: ArrayDataset):
        from flxbc.model import build_torch_model

        model = build_torch_model(tuple(dataset.images.shape[1:]), dataset.num_classes)
        return model

    def _tensor_images(self, dataset: ArrayDataset):
        torch = self.torch
        images = torch.tensor(dataset.images, dtype=torch.float32)
        if images.ndim == 4:
            images = images.unsqueeze(1)
        elif images.ndim == 3:
            images = images.unsqueeze(1)
        elif images.ndim == 5 and images.shape[-1] in (1, 3):
            images = images.permute(0, 4, 1, 2, 3)
        return images

    def _state_to_numpy(self, state) -> ParameterDict:
        return {name: value.detach().cpu().numpy().copy() for name, value in state.items()}

    def _numpy_to_state(self, parameters: ParameterDict):
        torch = self.torch
        return {name: torch.tensor(value) for name, value in parameters.items()}


def _classification_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    *,
    probabilities: np.ndarray | None = None,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    probabilities = _normalize_probabilities(predictions, num_classes, probabilities)
    accuracy = float(np.mean(predictions == labels)) if len(labels) else 0.0
    f1_values = []
    for class_id in range(num_classes):
        tp = float(np.sum((predictions == class_id) & (labels == class_id)))
        fp = float(np.sum((predictions == class_id) & (labels != class_id)))
        fn = float(np.sum((predictions != class_id) & (labels == class_id)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
    return {
        "loss": _cross_entropy_loss(probabilities, labels),
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_values)),
        "auc": _binary_auc(probabilities[:, 1], labels) if num_classes == 2 else accuracy,
    }


def _parameter_distance(left: ParameterDict, right: ParameterDict) -> float:
    total = 0.0
    for name in left:
        diff = left[name].astype(np.float64) - right[name].astype(np.float64)
        total += float(np.sum(diff * diff))
    return float(np.sqrt(total))


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def _normalize_probabilities(
    predictions: np.ndarray,
    num_classes: int,
    probabilities: np.ndarray | None,
) -> np.ndarray:
    if probabilities is not None:
        probabilities = np.asarray(probabilities, dtype=np.float64)
        clipped = np.clip(probabilities, 1e-12, 1.0)
        return clipped / clipped.sum(axis=1, keepdims=True)
    one_hot = np.full((len(predictions), num_classes), 1e-12, dtype=np.float64)
    if len(predictions):
        one_hot[np.arange(len(predictions)), predictions] = 1.0
    return one_hot / one_hot.sum(axis=1, keepdims=True)


def _cross_entropy_loss(probabilities: np.ndarray, labels: np.ndarray) -> float:
    if len(labels) == 0:
        return 0.0
    selected = probabilities[np.arange(len(labels)), labels]
    return -float(np.mean(np.log(np.clip(selected, 1e-12, 1.0))))


def _binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return 0.5
    comparisons = positives[:, None] - negatives[None, :]
    wins = float(np.sum(comparisons > 0))
    ties = float(np.sum(comparisons == 0))
    return (wins + 0.5 * ties) / float(len(positives) * len(negatives))
