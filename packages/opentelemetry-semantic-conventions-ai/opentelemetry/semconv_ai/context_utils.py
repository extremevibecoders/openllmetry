"""
Utility functions for safe OpenTelemetry context management.
"""


def safe_detach_context(token):
    """
    Safely detach context token without causing application crashes.

    This method implements a fail-safe approach to context detachment that handles
    all known edge cases in async/concurrent scenarios where context tokens may
    become invalid or be detached in different execution contexts.

    We use the runtime context directly to avoid logging errors from context_api.detach()
    
    Args:
        token: The context token to detach, or None
    """
    if not token:
        return

    try:
        # Use the runtime context directly to avoid error logging from context_api.detach()
        from opentelemetry.context import _RUNTIME_CONTEXT

        _RUNTIME_CONTEXT.detach(token)
    except Exception:
        # Context detach can fail in async scenarios when tokens are created in different contexts
        # This includes ValueError, RuntimeError, and other context-related exceptions
        # This is expected behavior and doesn't affect the correct span hierarchy
        #
        # Common scenarios where this happens:
        # 1. Token created in one async task/thread, detached in another
        # 2. Context was already detached by another process
        # 3. Token became invalid due to context switching
        # 4. Race conditions in highly concurrent scenarios
        #
        # This is safe to ignore as the span itself was properly ended
        # and the tracing data is correctly captured.
        pass
