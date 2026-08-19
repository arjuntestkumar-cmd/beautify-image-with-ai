"""Typed errors that map onto HTTP status codes and a stable `code` for the UI."""
from __future__ import annotations


class AppError(Exception):
    status_code = 500
    code = "INTERNAL_ERROR"

    def __init__(self, message: str = "Something went wrong.") -> None:
        super().__init__(message)
        self.message = message


class UnsupportedImageFormat(AppError):
    status_code = 415
    code = "UNSUPPORTED_FORMAT"


class CorruptedImage(AppError):
    status_code = 400
    code = "CORRUPTED_IMAGE"


class FileTooLarge(AppError):
    status_code = 413
    code = "FILE_TOO_LARGE"


class InputPixelLimitExceeded(AppError):
    status_code = 413
    code = "INPUT_TOO_LARGE"


class OutputPixelLimitExceeded(AppError):
    status_code = 422
    code = "OUTPUT_TOO_LARGE"


class QueueFull(AppError):
    status_code = 429
    code = "QUEUE_FULL"


class JobNotFound(AppError):
    status_code = 404
    code = "NOT_FOUND"


class ResultExpired(AppError):
    status_code = 410
    code = "EXPIRED"


class ModelsUnavailable(AppError):
    status_code = 503
    code = "MODELS_UNAVAILABLE"


class OutOfMemory(AppError):
    status_code = 503
    code = "OUT_OF_MEMORY"


class ProcessingFailed(AppError):
    status_code = 500
    code = "PROCESSING_FAILED"


class ProcessingTimeout(AppError):
    status_code = 504
    code = "TIMEOUT"
