"""Custom exceptions for Tierra+."""


class IntegrationNotConfiguredError(RuntimeError):
    """Raised when a backend or required dataset has not been configured."""
