# Fix for get_final_message AttributeError in opentelemetry-instrumentation-anthropic

## Problem
The current implementation (v0.46.2) of `opentelemetry-instrumentation-anthropic` wraps streaming responses from the Anthropic SDK with plain async generators that don't preserve the original methods from the stream objects. This causes an `AttributeError: 'async_generator' object has no attribute 'get_final_message'` when trying to access the `get_final_message` method that is available on the original Anthropic SDK stream objects.

## Root Cause
The issue occurs in two places:

1. **Direct streaming responses**: When `is_streaming_response(response)` returns `True`, the code calls `abuild_from_streaming_response()` which returns a plain `async_generator` that only yields events but doesn't expose the original stream's methods.

2. **Stream managers**: Similar issue with `WrappedAsyncMessageStreamManager` which also returns plain generators.

## Solution
Created wrapper classes that preserve all original methods while maintaining instrumentation functionality:

### 1. WrappedAsyncStream
- Wraps the original `AsyncStream` object
- Uses `__getattr__` delegation to preserve all original methods (including `get_final_message`)
- Implements `__aiter__` and `__anext__` to provide instrumented iteration
- Maintains full compatibility with the original stream API

### 2. WrappedStream  
- Similar wrapper for synchronous `Stream` objects
- Uses `__getattr__` delegation for method preservation
- Implements `__iter__` and `__next__` for instrumented iteration

### 3. Updated WrappedAsyncMessageStreamManager
- Now returns `WrappedAsyncStream` instead of plain generators
- Preserves all original functionality while adding instrumentation

## Files Modified

### `/workspace/packages/opentelemetry-instrumentation-anthropic/opentelemetry/instrumentation/anthropic/streaming.py`
- Added `WrappedAsyncStream` class
- Added `WrappedStream` class  
- Updated `WrappedAsyncMessageStreamManager` to use new wrapper
- Updated `WrappedMessageStreamManager` to use new wrapper

### `/workspace/packages/opentelemetry-instrumentation-anthropic/opentelemetry/instrumentation/anthropic/__init__.py`
- Updated async streaming response handler to use `WrappedAsyncStream`
- Updated sync streaming response handler to use `WrappedStream`

## Key Features of the Fix

1. **Full Method Preservation**: All methods from the original stream objects are preserved via `__getattr__` delegation
2. **Instrumentation Maintained**: All existing telemetry, metrics, and tracing functionality is preserved
3. **Backward Compatibility**: Existing code continues to work without changes
4. **Performance**: Minimal overhead - delegation only happens for method access, not iteration

## Testing
The fix has been tested with mock objects to verify:
- `get_final_message` method is accessible on wrapped streams
- Other methods are also preserved
- Streaming iteration continues to work
- Instrumentation functionality is maintained

## Impact
This fix resolves the `AttributeError` while maintaining full backward compatibility and all existing instrumentation features. Users can now access `get_final_message` and any other methods available on the original Anthropic SDK stream objects.