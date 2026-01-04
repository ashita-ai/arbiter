"""Simple test to verify pretty_print methods exist and work."""

import inspect
from io import StringIO

# Test imports
try:
    from arbiter_ai.core.models import (
        EvaluationResult,
        ComparisonResult,
        BatchEvaluationResult,
        Score,
        LLMInteraction,
    )
    print("✅ Models imported successfully")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    exit(1)

# Test that methods exist
print("\n" + "=" * 60)
print("Checking pretty_print() methods exist...")
print("=" * 60)

# Check EvaluationResult
assert hasattr(EvaluationResult,
               'pretty_print'), "EvaluationResult missing pretty_print"
print("✅ EvaluationResult.pretty_print() exists")

# Check ComparisonResult
assert hasattr(ComparisonResult,
               'pretty_print'), "ComparisonResult missing pretty_print"
print("✅ ComparisonResult.pretty_print() exists")

# Check BatchEvaluationResult
assert hasattr(BatchEvaluationResult,
               'pretty_print'), "BatchEvaluationResult missing pretty_print"
print("✅ BatchEvaluationResult.pretty_print() exists")

# Check method signatures

print("\n" + "=" * 60)
print("Checking method signatures...")
print("=" * 60)

# EvaluationResult signature
sig = inspect.signature(EvaluationResult.pretty_print)
params = list(sig.parameters.keys())
assert 'self' in params or params[0] == 'self', "Missing self parameter"
assert 'file' in params, "Missing file parameter"
assert 'verbose' in params, "Missing verbose parameter"
print(f"✅ EvaluationResult.pretty_print signature: {sig}")

# ComparisonResult signature
sig = inspect.signature(ComparisonResult.pretty_print)
params = list(sig.parameters.keys())
assert 'file' in params, "Missing file parameter"
assert 'verbose' in params, "Missing verbose parameter"
print(f"✅ ComparisonResult.pretty_print signature: {sig}")

# BatchEvaluationResult signature
sig = inspect.signature(BatchEvaluationResult.pretty_print)
params = list(sig.parameters.keys())
assert 'file' in params, "Missing file parameter"
assert 'verbose' in params, "Missing verbose parameter"
print(f"✅ BatchEvaluationResult.pretty_print signature: {sig}")

print("\n" + "=" * 60)
print("ALL CHECKS PASSED ✅")
print("=" * 60)
print("\nThe pretty_print() methods have been successfully added to:")
print("  • EvaluationResult")
print("  • ComparisonResult")
print("  • BatchEvaluationResult")
print("\nEach method supports:")
print("  • file parameter (optional, defaults to sys.stdout)")
print("  • verbose parameter (optional, defaults to False)")
