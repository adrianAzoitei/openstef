# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project
#
# SPDX-License-Identifier: MPL-2.0

"""KServe / Triton inference backend.

Executes foundation models through a KServe endpoint instead of running locally.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Self

import numpy as np

from openstef_foundation_models.models.checkpoint import CheckpointMetadata


class KServeBackend:
    """An InferenceBackend backed by a KServe API endpoint.

    Sends named input tensors to a KServe v2 endpoint (REST or gRPC) and
    receives named outputs. The checkpoint metadata is loaded locally from
    the `.metadata.json` file (no weights needed).
    """

    def __init__(self, metadata: CheckpointMetadata, model_url: str) -> None:
        """Initialize a KServe backend.

        Args:
            metadata: Checkpoint metadata (input/output names, shapes, quantiles, etc.).
            model_url: url of the kserve inference service.
        """
        self._metadata = metadata
        self.model_url = model_url
        self._client = None

    @classmethod
    def from_metadata_path(cls, metadata_path: Path, model_url: str) -> Self:
        """Load metadata from a local .metadata.json file and create a backend.

        Args:
            metadata_path: Path to the checkpoint's .metadata.json file.
            model_url: url of the kserve inference service.

        Returns:
            A KServeBackend ready to use.
        """
        metadata = CheckpointMetadata.model_validate_json(Path(metadata_path).read_text(encoding="utf-8"))
        return cls(metadata=metadata, model_url=model_url)

    @property
    def metadata(self) -> CheckpointMetadata:
        """Metadata describing the checkpoint this backend executes."""
        return self._metadata

    def run(self, inputs: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
        """Execute the model on a batch of named input tensors via KServe.

        Args:
            inputs: Named input tensors. Keys must match ``metadata.input_names``.

        Returns:
            Named output tensors, including ``metadata.output_name``.

        Raises:
            RuntimeError: If the KServe request fails.
        """
        # TODO: Convert inputs to KServe v2 / OpenInference Protocol format.
        # TODO: Send via httpx or tritonclient.
        # TODO: Parse response and convert back to {name: np.ndarray}.
        raise NotImplementedError("KServe tensor transformation and HTTP dispatch not yet implemented.")

    def close(self) -> None:
        """Release any resources held by the backend (e.g., HTTP client)."""
        if self._client is not None:
            self._client = None


__all__ = ["KServeBackend"]
