# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project
#
# SPDX-License-Identifier: MPL-2.0

"""KServe / Triton inference backend.

Executes foundation models through a KServe endpoint instead of running locally.
Speaks the v2 (Open Inference) REST protocol with JSON-encoded tensors.

Requires the ``[kserve]`` extra: ``pip install openstef-foundation-models[kserve]``
"""

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Self

import numpy as np

from openstef_core.exceptions import MissingExtraError
from openstef_foundation_models.models.checkpoint import CheckpointMetadata

try:
    import httpx
except ImportError as e:
    raise MissingExtraError("httpx", "openstef-foundation-models", install_extra="kserve") from e

_NUMPY_TO_V2: dict[str, str] = {
    "bool": "BOOL",
    "int8": "INT8",
    "int16": "INT16",
    "int32": "INT32",
    "int64": "INT64",
    "uint8": "UINT8",
    "uint16": "UINT16",
    "uint32": "UINT32",
    "uint64": "UINT64",
    "float16": "FP16",
    "float32": "FP32",
    "float64": "FP64",
}
_V2_TO_NUMPY: dict[str, str] = {v2: name for name, v2 in _NUMPY_TO_V2.items()}

BearerToken = str | Callable[[], str]


def _encode_tensor(name: str, array: np.ndarray) -> dict[str, Any]:
    """Encode a numpy array as a v2 protocol JSON input tensor.

    Args:
        name: Tensor name.
        array: Tensor values.

    Returns:
        The v2 tensor object with a flattened, row-major ``data`` list.

    Raises:
        ValueError: If the array dtype has no v2 equivalent.
    """
    datatype = _NUMPY_TO_V2.get(array.dtype.name)
    if datatype is None:
        msg = f"Input '{name}' has unsupported dtype {array.dtype}; supported: {sorted(_NUMPY_TO_V2)}"
        raise ValueError(msg)
    return {"name": name, "shape": list(array.shape), "datatype": datatype, "data": array.ravel().tolist()}


def _decode_tensor(tensor: Mapping[str, Any]) -> tuple[str, np.ndarray]:
    """Decode a v2 protocol JSON output tensor into a numpy array.

    Args:
        tensor: The v2 tensor object (``name``, ``shape``, ``datatype``, ``data``).

    Returns:
        The tensor name and its values as an array of the declared shape.

    Raises:
        ValueError: If the datatype is unsupported.
    """
    dtype = _V2_TO_NUMPY.get(str(tensor["datatype"]).upper())
    if dtype is None:
        msg = f"Output '{tensor['name']}' has unsupported datatype {tensor['datatype']}"
        raise ValueError(msg)
    return tensor["name"], np.asarray(tensor["data"], dtype=dtype).reshape(tensor["shape"])


class KServeBackend:
    """An InferenceBackend backed by a KServe v2 REST endpoint.

    Sends named input tensors to the endpoint as JSON and returns the named
    output tensors. The checkpoint metadata is loaded locally from the
    ``.metadata.json`` file (no weights needed).
    """

    def __init__(
        self,
        metadata: CheckpointMetadata,
        model_url: str,
        bearer_token: BearerToken | None = None,
        timeout: float = 60.0,
    ) -> None:
        """Initialize a KServe backend.

        Args:
            metadata: Checkpoint metadata (input/output names, shapes, quantiles, etc.).
            model_url: Full inference URL of the model, including scheme, used as-is
                (e.g. ``https://host/v2/models/chronos2/infer``).
            bearer_token: Token sent as ``Authorization: Bearer <token>``. Pass a callable to
                have it invoked on every request, so the caller can handle token refresh.
                This library never fetches tokens itself.
            timeout: Request timeout in seconds.
        """
        self._metadata = metadata
        self.model_url = model_url
        self._bearer_token = bearer_token
        self._timeout = timeout
        self._client: httpx.Client | None = None

    @classmethod
    def from_metadata_path(
        cls,
        metadata_path: Path,
        model_url: str,
        bearer_token: BearerToken | None = None,
        timeout: float = 60.0,
    ) -> Self:
        """Load metadata from a local .metadata.json file and create a backend.

        Args:
            metadata_path: Path to the checkpoint's .metadata.json file.
            model_url: Full inference URL of the model, including scheme.
            bearer_token: Token (or token-returning callable) for the ``Authorization`` header.
            timeout: Request timeout in seconds.

        Returns:
            A KServeBackend ready to use.
        """
        metadata = CheckpointMetadata.model_validate_json(Path(metadata_path).read_text(encoding="utf-8"))
        return cls(metadata=metadata, model_url=model_url, bearer_token=bearer_token, timeout=timeout)

    @property
    def metadata(self) -> CheckpointMetadata:
        """Metadata describing the checkpoint this backend executes."""
        return self._metadata

    def _headers(self) -> dict[str, str]:
        if self._bearer_token is None:
            return {}
        token = self._bearer_token() if callable(self._bearer_token) else self._bearer_token
        return {"Authorization": f"Bearer {token}"}

    def run(self, inputs: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
        """Execute the model on a batch of named input tensors via KServe.

        Args:
            inputs: Named input tensors. Keys must match ``metadata.input_names``.

        Returns:
            Named output tensors exactly as returned by the server.

        Raises:
            RuntimeError: If the KServe request fails or the response is malformed.
        """
        body = {"inputs": [_encode_tensor(name, array) for name, array in inputs.items()]}

        if self._client is None:
            self._client = httpx.Client(timeout=self._timeout)
        try:
            response = self._client.post(self.model_url, json=body, headers=self._headers())
        except httpx.HTTPError as e:
            msg = f"KServe request to {self.model_url} failed: {e}"
            raise RuntimeError(msg) from e

        if response.is_error:
            msg = f"KServe request to {self.model_url} failed with {response.status_code}: {response.text[:500]}"
            raise RuntimeError(msg)

        try:
            tensors = response.json()["outputs"]
            return dict(_decode_tensor(tensor) for tensor in tensors)
        except (ValueError, KeyError, TypeError) as e:
            msg = f"Malformed KServe response from {self.model_url}: {e}; body: {response.text[:500]}"
            raise RuntimeError(msg) from e

    def close(self) -> None:
        """Release any resources held by the backend (e.g., HTTP client)."""
        if self._client is not None:
            self._client.close()
            self._client = None


__all__ = ["BearerToken", "KServeBackend"]
