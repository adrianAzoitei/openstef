# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Presets for building foundation-model forecasting workflows from config.

A :class:`ForecastingWorkflowConfig` declares the model family, the checkpoint
that backs it, the requested quantiles/horizons, the target column, and the
columns to keep; :func:`create_forecasting_workflow` turns it into a
:class:`~openstef_models.workflows.custom_forecasting_workflow.CustomForecastingWorkflow`
with feature-selection preprocessing and quantile-sorting postprocessing. Every
selected non-target column is forwarded to the model as a known covariate.

The checkpoint defaults to the published OpenSTEF Chronos-2 model on the
HuggingFace Hub, so the minimal config is just::

    workflow = create_forecasting_workflow(ForecastingWorkflowConfig())

Pick a different size or variant through :class:`~openstef_foundation_models.models.catalog.Chronos2`,
or pass a :class:`~openstef_foundation_models.models.checkpoint.LocalCheckpoint`
to run a file already on disk.
"""

from pathlib import Path
from typing import Literal, assert_never

from pydantic import Field, SecretStr

from openstef_core.base_model import BaseConfig
from openstef_core.mixins import TransformPipeline
from openstef_core.types import LeadTime, Q, Quantile
from openstef_foundation_models.inference.backend import InferenceBackend
from openstef_foundation_models.inference.provider_selection import DefaultProviderPolicy
from openstef_foundation_models.inference.providers import ExecutionProvider, SessionOptionsConfig
from openstef_foundation_models.models.catalog import Chronos2
from openstef_foundation_models.models.checkpoint import CheckpointRef
from openstef_foundation_models.models.forecasting.chronos2_forecaster import (
    Chronos2Forecaster,
    Chronos2HyperParams,
)
from openstef_models.mixins import ModelIdentifier
from openstef_models.models import ForecastingModel
from openstef_models.transforms.general import Selector
from openstef_models.transforms.postprocessing import QuantileSorter
from openstef_models.utils.feature_selection import FeatureSelection
from openstef_models.workflows.custom_forecasting_workflow import (
    CustomForecastingWorkflow,
    ForecastingCallback,
)


class OnnxBackendConfig(BaseConfig):
    """Compute configuration for an ONNX Runtime inference backend.

    Holds only *how* to run the model (execution providers, session options), not
    *which* weights: the checkpoint is supplied to :meth:`build` by the caller, so
    the same compute settings can run different checkpoints.
    """

    kind: Literal["onnx"] = Field(default="onnx", description="Discriminator tag for backend type.")
    providers: list[ExecutionProvider] | None = Field(
        default=None,
        description="Ordered execution providers to try. ``None`` lets :attr:`policy` pick a host-appropriate "
        "chain from the checkpoint metadata (graceful). An explicit list is used exactly as given (strict: a "
        "missing accelerator raises).",
    )
    policy: DefaultProviderPolicy = Field(
        default=DefaultProviderPolicy(),
        description="Selection policy used when :attr:`providers` is None. Maps the checkpoint's precision and "
        "static-shape-ness plus the host to an ordered provider chain. Replace it (e.g. a subclass overriding "
        "select) to target hardware the default does not cover.",
    )
    session_options: SessionOptionsConfig | None = Field(
        default=None,
        description="Optional ONNX Runtime session options.",
    )

    def build(self, checkpoint: CheckpointRef) -> InferenceBackend:
        """Resolve *checkpoint* and build the ONNX Runtime backend.

        Importing the backend raises ``MissingExtraError`` if ONNX Runtime is not installed.

        Args:
            checkpoint: The checkpoint (weights + metadata) to load and run.

        Returns:
            A ready-to-run backend wrapping the resolved checkpoint.
        """
        from openstef_foundation_models.inference.onnx_backend import OnnxBackend  # noqa: PLC0415

        resolved = checkpoint.resolve()
        return OnnxBackend.from_checkpoint(
            resolved,
            providers=self.providers,
            session_options=self.session_options,
            policy=self.policy,
        )


class KServeBackendConfig(BaseConfig):
    """Configuration for a KServe inference backend.

    Holds KServe-specific connection details instead of local ONNX Runtime settings.
    """

    kind: Literal["kserve"] = Field(default="kserve", description="Discriminator tag for backend type.")
    metadata_path: Path = Field(description="Local path to the checkpoint's .metadata.json file (no weights needed).")
    model_url: str = Field(
        description="Full inference URL of the model, including scheme, used as-is "
        "(e.g. 'https://host/v2/models/chronos2/infer').",
    )
    bearer_token: SecretStr | None = Field(
        default=None,
        description="Optional token sent as 'Authorization: Bearer <token>'. Supplied by the caller; never fetched "
        "here. For tokens that expire, construct KServeBackend directly with a callable instead.",
    )

    def build(self, checkpoint: CheckpointRef) -> InferenceBackend:  # noqa: ARG002
        """Build a KServeBackend from this config.

        Note: The checkpoint argument is ignored; KServeBackendConfig only uses
        the local metadata_path to load checkpoint metadata.

        Args:
            checkpoint: Unused; required by the BackendConfig interface.

        Returns:
            A ready-to-use KServeBackend.
        """
        from openstef_foundation_models.inference.kserve_backend import KServeBackend  # noqa: PLC0415

        bearer_token = self.bearer_token.get_secret_value() if self.bearer_token else None
        return KServeBackend.from_metadata_path(
            metadata_path=self.metadata_path,
            model_url=self.model_url,
            bearer_token=bearer_token,
        )


#: A backend configuration. Supports both local ONNX Runtime (``OnnxBackendConfig``)
#: and remote KServe (``KServeBackendConfig``) backends. Kept as a named type so new
#: backends can be added without changing the workflow config's public signature.
BackendConfig = OnnxBackendConfig | KServeBackendConfig


class ForecastingWorkflowConfig(BaseConfig):
    """Declarative configuration for a foundation-model forecasting workflow.

    Selects a model family and the checkpoint that backs it, the requested
    quantiles and horizons, the target column, and the columns to keep before
    forecasting. Every kept non-target column is treated as a known covariate, so
    weather forecasts condition the prediction. Compute settings (execution
    providers, session options) live on the nested :attr:`backend` config.
    """

    model: Literal["chronos2"] = Field(default="chronos2", description="Foundation model family to use.")
    checkpoint: CheckpointRef = Field(
        default=Chronos2.BASE.checkpoint(),
        description="Checkpoint (weights + metadata) to load and run. Defaults to the published OpenSTEF "
        "Chronos-2 dynamic ONNX checkpoint on the HuggingFace Hub. Pick a size and variant with "
        "`Chronos2.<SIZE>.checkpoint(...)`, or pass a LocalCheckpoint to run a file already on disk.",
    )

    quantiles: list[Quantile] = Field(
        default=[Q(0.5)],
        min_length=1,
        description="Quantile levels to predict.",
    )
    horizons: list[LeadTime] = Field(
        default=[LeadTime.from_string("PT48H")],
        min_length=1,
        description="Forecast horizons to predict.",
    )

    target_column: str = Field(default="load", description="Name of the target column to forecast.")
    selected_features: FeatureSelection = Field(
        default_factory=lambda: FeatureSelection.ALL,
        description="Columns to keep before forecasting. Defaults to all columns. Every kept non-target column "
        "is forwarded to the model as a known covariate. Must include the target column.",
    )

    backend: BackendConfig = Field(
        default_factory=OnnxBackendConfig,
        description="Inference backend (compute) configuration: execution providers and session options.",
    )

    chronos2_hyperparams: Chronos2HyperParams = Field(
        default_factory=Chronos2HyperParams,
        description="Hyperparameters for the Chronos-2 forecaster.",
    )

    model_id: ModelIdentifier = Field(default="chronos2", description="Unique identifier for the workflow model.")
    run_name: str | None = Field(default=None, description="Optional name for this workflow run.")
    experiment_tags: dict[str, str] = Field(
        default_factory=dict,
        description="Optional metadata tags for experiment tracking.",
    )


def create_forecasting_workflow(config: ForecastingWorkflowConfig) -> CustomForecastingWorkflow:
    """Build a foundation-model forecasting workflow from a declarative config.

    Resolves the checkpoint (lazily importing the inference runtime), composes
    the forecaster for the selected model family on the configured backend, and
    wraps it in a workflow with feature-selection preprocessing and
    quantile-sorting postprocessing. There is no training step: the model is
    zero-shot, so :meth:`CustomForecastingWorkflow.fit` only fits the feature
    selector.

    Args:
        config: The workflow configuration.

    Returns:
        A ready-to-use workflow composing the configured backend.
    """
    match config.model:
        case "chronos2":
            model_backend = config.backend.build(config.checkpoint)
            forecaster = Chronos2Forecaster(
                backend=model_backend,
                quantiles=config.quantiles,
                horizons=config.horizons,
                hyperparams=config.chronos2_hyperparams,
            )
        case _:
            assert_never(config.model)

    callbacks: list[ForecastingCallback] = []
    return CustomForecastingWorkflow(
        model=ForecastingModel(
            preprocessing=TransformPipeline(transforms=[Selector(selection=config.selected_features)]),
            forecaster=forecaster,
            postprocessing=TransformPipeline(transforms=[QuantileSorter()]),
            target_column=config.target_column,
        ),
        model_id=config.model_id,
        run_name=config.run_name,
        callbacks=callbacks,
        experiment_tags=config.experiment_tags,
    )


__all__ = [
    "BackendConfig",
    "ForecastingWorkflowConfig",
    "KServeBackendConfig",
    "OnnxBackendConfig",
    "create_forecasting_workflow",
]
