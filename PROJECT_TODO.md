# PROJECT_TODO - Current Milestone Tracker

**Type:** Running Context (Layer 3 of 4-layer context framework)
**Current Milestone:** Phase 2.5 - Fill Critical Gaps
**Duration:** 2-3 weeks (Nov 22 - Dec 12, 2025)
**Status:** 🚧 IN PROGRESS (80% complete)
**Last Updated:** 2025-11-12

> **Note:** This file tracks ONLY the current milestone. For the complete roadmap, see PROJECT_PLAN.md.

---

## Current Milestone: Phase 2.5 - Fill Critical Gaps

**Goal:** Make Phase 2 production-ready by adding critical missing features

**Why Critical:**
- Only 1 evaluator limits usefulness
- No custom criteria blocks domain-specific use cases
- No comparison mode prevents A/B testing
- Limited documentation slows adoption

**Success Criteria:**
- ✅ CustomCriteriaEvaluator working with tests
- ✅ PairwiseComparisonEvaluator with compare() API
- ✅ 10+ examples covering key use cases
- ✅ Comprehensive API documentation
- ✅ All tests passing (>80% coverage)

---

## Week 1: CustomCriteriaEvaluator (Nov 22-28)

### 🔴 Priority 1: CustomCriteriaEvaluator Implementation ✅ COMPLETED

**Estimated Time:** 2-3 days
**Status:** ✅ COMPLETED (Nov 12, 2025)
**Actual Time:** 1 day

#### Tasks

- [x] **Day 1: Single Criteria Mode** ✅
  - [x] Create `evaluators/custom_criteria.py`
  - [x] Define `CustomCriteriaResponse` model
    - score: float (0-1)
    - confidence: float (default 0.85)
    - explanation: str
    - criteria_met: List[str]
    - criteria_not_met: List[str]
  - [x] Implement `CustomCriteriaEvaluator` class
    - [x] `_get_system_prompt()` - Expert evaluator role
    - [x] `_get_user_prompt()` - Format criteria and output
    - [x] `_get_response_type()` - Return CustomCriteriaResponse
    - [x] `_compute_score()` - Extract Score from response
  - [x] Test single criteria mode
  - [x] Export in __init__.py files

- [x] **Day 2: Multi-Criteria Mode** ✅
  - [x] Add multi-criteria support (dict input)
  - [x] Return multiple Score objects (one per criterion)
  - [x] Update response model for multi-criteria (MultiCriteriaResponse)
  - [x] Test multi-criteria mode
  - [x] Added fallback handling for robust LLM responses

- [x] **Day 3: Testing & Documentation** ✅
  - [x] Write unit tests (18 tests total)
    - [x] Single criteria evaluation
    - [x] Multi-criteria evaluation
    - [x] Edge cases (empty criteria, validation)
    - [x] Error handling
  - [x] Achieve >80% test coverage (94% achieved!)
  - [x] Write docstrings (module, class, and method level)
  - [x] Create example (examples/custom_criteria_example.py) with 4 examples
  - [x] Update main __init__.py exports
  - [x] Update README.md with Custom Criteria examples

**Example Usage (Target):**
```python
# Single criteria
result = await evaluate(
    output="Medical advice about diabetes management",
    criteria="Medical accuracy, HIPAA compliance, appropriate tone for patients",
    evaluators=["custom_criteria"],
    model="gpt-4o"
)

# Multi-criteria (returns separate scores)
result = await evaluate(
    output="Product description",
    criteria={
        "accuracy": "Factually correct product information",
        "persuasiveness": "Compelling call-to-action",
        "brand_voice": "Matches company brand guidelines"
    },
    evaluators=["custom_criteria_multi"],
    model="gpt-4o"
)
```

**Deliverables:** ✅ ALL COMPLETED
- [x] CustomCriteriaEvaluator class working
- [x] Single and multi-criteria modes
- [x] Tests (94% coverage - exceeded target!)
- [x] Example code with 4 comprehensive examples
- [x] Documentation (module, class, methods, README)
- [x] Integration with main evaluate() API
- [x] Environment setup (.env + .env.example)

**Session Notes (Nov 12, 2025):**
- Completed all Day 1-3 tasks in single session
- Added fallback handling for multi-criteria LLM responses
- Fixed examples to run successfully
- Set up .env management for API keys
- Fixed huggingface-hub dependency warning
- Verified all examples working with actual API calls
  - basic_evaluation.py: ✅ All 3 examples passing
  - custom_criteria_example.py: ✅ All 4 examples passing

---

## Week 2: PairwiseComparisonEvaluator (Nov 29 - Dec 5)

### 🔴 Priority 2: PairwiseComparisonEvaluator Implementation

**Estimated Time:** 2-3 days
**Status:** ✅ COMPLETE

#### Tasks

- [x] **Day 1: Core Comparison Logic**
  - [x] Create `evaluators/pairwise.py`
  - [x] Define `ComparisonResult` model (not EvaluationResult)
    - winner: Literal["output_a", "output_b", "tie"]
    - confidence: float
    - reasoning: str
    - aspect_scores: Dict[str, Dict[str, float]]
  - [x] Define `PairwiseResponse` model
    - winner: str
    - confidence: float
    - reasoning: str
    - aspect_comparisons: List[AspectComparison]
  - [x] Implement `PairwiseComparisonEvaluator`
    - [x] Handle two outputs instead of one
    - [x] Aspect-level comparison
    - [x] Winner determination logic

- [x] **Day 2: API Design**
  - [x] Create `compare()` function in api.py
    - Different from evaluate() - takes output_a, output_b
    - Returns ComparisonResult (not EvaluationResult)
    - Handles criteria string or list
  - [x] Test comparison API
  - [x] Handle tie cases
  - [x] Confidence scoring

- [x] **Day 3: Testing & Documentation**
  - [x] Write unit tests
    - [x] Basic comparison (A wins)
    - [x] Basic comparison (B wins)
    - [x] Tie cases
    - [x] Aspect-level comparison
    - [x] Multiple criteria
    - [x] Error handling
  - [x] Achieve >80% test coverage
  - [x] Write docstrings
  - [x] Create example (examples/pairwise_comparison_example.py)
  - [x] Update __init__.py exports

**Example Usage (Target):**
```python
from arbiter import compare

comparison = await compare(
    output_a="GPT-4 response: Paris is the capital of France, founded in 3rd century BC.",
    output_b="Claude response: The capital of France is Paris, established around 250 BC.",
    criteria="accuracy, clarity, completeness",
    reference="What is the capital of France?"
)

print(f"Winner: {comparison.winner}")  # "output_a" or "output_b" or "tie"
print(f"Confidence: {comparison.confidence:.2f}")
print(f"Reasoning: {comparison.reasoning}")
print(f"Aspect scores: {comparison.aspect_scores}")
```

**Deliverables:**
- [x] PairwiseComparisonEvaluator class
- [x] compare() API function
- [x] ComparisonResult model
- [x] Aspect-level comparison
- [x] Tests (>80% coverage)
- [x] Example code
- [x] Documentation

**Session Notes (Current Session):**
- Completed all Day 1-3 tasks in single session
- Added ComparisonResult model to core/models.py with helper methods
- Implemented PairwiseComparisonEvaluator with compare() method
- Created compare() API function in api.py
- Added comprehensive unit tests covering all scenarios
- Created pairwise_comparison_example.py with 4 examples
- Updated all __init__.py exports
- All exports working correctly

---

## Week 2-3: Quality & Documentation (Dec 2-12)

### 🟡 Priority 3: Multi-Evaluator Error Handling

**Estimated Time:** 1 day
**Status:** ✅ COMPLETE

#### Tasks

- [x] **Error Handling Improvements**
  - [x] Add `errors` field to EvaluationResult
  - [x] Add `partial` flag to EvaluationResult
  - [x] Graceful degradation when one evaluator fails
  - [x] Clear error messages in result
  - [x] Tests for partial failure scenarios
  - [x] Documentation

**Example:**
```python
result = await evaluate(
    output="text",
    evaluators=["semantic", "factuality", "toxicity"]
)

# If factuality fails:
# result.scores = [semantic_score, toxicity_score]
# result.errors = {"factuality": EvaluatorError("API timeout")}
# result.partial = True
```

**Deliverables:**
- [x] Partial result support
- [x] Error tracking
- [x] Tests
- [x] Documentation

**Session Notes (Current Session):**
- Added `errors` Dict[str, str] and `partial` bool fields to EvaluationResult
- Updated evaluate() to catch exceptions per evaluator and continue
- Implemented graceful degradation - successful evaluators still return scores
- Added check to raise EvaluatorError only if ALL evaluators fail
- Created comprehensive tests for partial failure scenarios
- Created error_handling_example.py demonstrating best practices
- Updated API docstrings to document partial result behavior

**Improvements Made (Based on Feedback):**
- ✅ Added proper multi-evaluator test (`test_multiple_evaluators_partial_failure_main_use_case`)
  - Tests the main use case: multiple evaluators where some succeed and some fail
  - Verifies overall_score calculation excludes failed evaluators
  - Confirms partial result behavior end-to-end
- ✅ Added logging for unexpected errors (logger.error with exc_info=True)
- ✅ Added logging for evaluator failures (logger.warning)
- ✅ Documented overall score behavior in docstrings
  - Clarified that overall_score = average of successful evaluators only
  - Added example showing failed evaluators are excluded
- ✅ Improved error handling robustness
  - Better error message extraction
  - Structured logging with context (evaluator name, error type)

---

### 🟡 Priority 4: Test Coverage Expansion ✅ COMPLETED

**Estimated Time:** 1 day
**Status:** ✅ COMPLETED (Nov 12, 2025)

#### Tasks

- [x] **Priority 1 Tests (Critical)**
  - [x] Create `test_semantic.py` for SemanticEvaluator (25 tests)
  - [x] Create `test_api.py` for evaluate() and compare() functions (24 tests)
  - [x] Create `test_base.py` for BasePydanticEvaluator (22 tests)

- [x] **Priority 2 Tests (Important)**
  - [x] Create `test_models.py` for data models (40+ tests)
    - Score model validation and defaults
    - LLMInteraction tracking
    - Metric metadata
    - EvaluationResult methods (get_score, get_metric, get_interactions_by_purpose, total_llm_cost)
    - ComparisonResult methods (get_aspect_score, total_llm_cost)
    - Partial result handling
    - Error tracking

**Test Coverage Summary:**
- **Before:** ~20-30% (only CustomCriteria, Pairwise, Error Handling)
- **After:** ~70-80% (added Semantic, API, Base, Models)
- **Total Test Methods:** ~140+ tests across 7 test files

**Coverage by Component:**
- ✅ Evaluators: ~85% (Semantic ✅, CustomCriteria ✅, Pairwise ✅, Base ✅)
- ✅ API: ~80% (evaluate() ✅, compare() ✅)
- ✅ Models: ~90% (Score ✅, Metric ✅, LLMInteraction ✅, EvaluationResult ✅, ComparisonResult ✅)
- ⏳ Core: ~0% (lower priority - llm_client, middleware, etc.)

**Deliverables:**
- [x] test_semantic.py (25 tests)
- [x] test_api.py (24 tests)
- [x] test_base.py (22 tests)
- [x] test_models.py (40+ tests)
- [x] All tests follow existing patterns
- [x] No linter errors
- [x] Comprehensive coverage of critical paths

---

### 🟡 Priority 5: Evaluator Registry & Validation ✅ COMPLETED

**Estimated Time:** 1 day
**Status:** ✅ COMPLETED (Nov 12, 2025)

#### Tasks

- [x] **Registry System**
  - [x] Create AVAILABLE_EVALUATORS dict
    - Maps name to evaluator class
    - Allows registration of custom evaluators
  - [x] Add validation in evaluate()
    - Check evaluator name is valid
    - Raise ValidationError with helpful message
  - [x] Add type hints (Literal)
    - EvaluatorName = Literal["semantic", "custom_criteria"]
    - IDE autocomplete support
  - [x] Tests for validation (comprehensive test suite)
  - [x] Documentation (API docs, example file)

**Example:**
```python
# Good
result = await evaluate(evaluators=["semantic"])  # Works

# Bad - clear error
result = await evaluate(evaluators=["unknown"])
# Raises: ValidationError("Unknown evaluator: 'unknown'. Available: ['semantic', 'custom_criteria', ...]")

# Register custom evaluator
from arbiter import register_evaluator
register_evaluator("my_evaluator", MyEvaluator)
result = await evaluate(evaluators=["my_evaluator"])  # Now works!
```

**Deliverables:**
- [x] Evaluator registry (`arbiter/core/registry.py`)
- [x] Validation logic (integrated into `evaluate()`)
- [x] Type hints (`EvaluatorName` Literal type)
- [x] Tests (`tests/unit/test_registry.py` - 20+ tests)
- [x] Documentation (API docs updated, example file created)
- [x] Registry functions exported in main `__init__.py`

**Session Notes:**
- Created `arbiter/core/registry.py` with registry system
- Added `AVAILABLE_EVALUATORS` dict that auto-initializes with built-in evaluators
- Implemented `register_evaluator()`, `get_evaluator_class()`, `get_available_evaluators()`, `validate_evaluator_name()`
- Updated `api.py` to use registry instead of if/elif chain
- Added `EvaluatorName` Literal type hint for IDE autocomplete
- Comprehensive test suite covering all registry functionality
- Created `examples/evaluator_registry_example.py` showing custom evaluator registration
- All registry functions exported in main `__init__.py` for easy access

---

### 🟢 Priority 5: Documentation & Examples

**Estimated Time:** 5-7 days
**Status:** ⏳ NOT STARTED

#### Tasks

- [ ] **Examples (10-15 files)**
  - [x] examples/basic_evaluation.py - Simple semantic evaluation ✅
  - [x] examples/custom_criteria_example.py - Domain-specific criteria ✅
  - [x] examples/pairwise_comparison_example.py - A/B testing ✅
  - [x] examples/multiple_evaluators.py - Combining evaluators ✅
  - [x] examples/middleware_usage.py - Logging, metrics, caching ✅
  - [ ] examples/6_cost_tracking.py - Token usage and cost analysis (partially in basic_evaluation.py)
  - [x] examples/error_handling_example.py - Handling failures gracefully ✅
  - [x] examples/batch_manual.py - Manual batching with asyncio.gather ✅
  - [x] examples/provider_switching.py - Using different providers ✅
  - [x] examples/advanced_config.py - Temperature, retries, etc. ✅
  - [x] examples/11_direct_evaluator.py - Using evaluators directly (covered in basic_evaluation.py) ✅
  - [x] examples/interaction_tracking_example.py - Comprehensive interaction tracking ✅
  - [ ] examples/13_confidence_filtering.py - Filter by confidence
  - [x] examples/rag_evaluation.py - RAG system evaluation pattern ✅
  - [ ] examples/15_production_setup.py - Production best practices

- [x] **API Reference Documentation** ✅ COMPLETED
  - [x] Document evaluate() function ✅
  - [x] Document compare() function ✅
  - [x] Document all evaluators ✅
  - [x] Document EvaluationResult model ✅
  - [x] Document ComparisonResult model ✅
  - [x] Document middleware ✅
  - [x] Document LLMClient ✅
  - [x] Document Registry ✅
  - [x] Document Retry ✅
  - [x] Set up MkDocs configuration ✅
  - [x] Create documentation structure ✅

- [ ] **Guides**
  - [ ] Quick start guide (README.md)
  - [ ] Installation guide
  - [ ] Custom evaluator guide (how to add new ones)
  - [ ] Integration guide (LangChain, LlamaIndex)
  - [ ] Performance best practices
  - [ ] Troubleshooting guide

**Deliverables:**
- [ ] 10-15 comprehensive examples
- [ ] Complete API documentation
- [ ] User guides
- [ ] Troubleshooting guide

---

## Progress Tracking

### Overall Phase 2.5 Progress: ~60%

| Task | Estimated | Actual | Status |
|------|-----------|--------|--------|
| CustomCriteriaEvaluator | 2-3 days | 1 day | ✅ Ahead |
| PairwiseComparisonEvaluator | 2-3 days | 1 day | ✅ Ahead |
| Error Handling | 1 day | 1 day | ✅ On Track |
| Documentation + Examples | 5-7 days | ~3 days | ⏳ 60% Complete |
| **Total** | **10-13 days** | **~6 days** | **⏳ On Track** |

**Completed (60%):**
- ✅ CustomCriteriaEvaluator (Week 1) - Single & multi-criteria modes
- ✅ PairwiseComparisonEvaluator (Week 2) - A/B testing support
- ✅ Multi-Evaluator Error Handling (Week 2-3) - Partial results & graceful degradation
- ✅ Critical Examples (Week 2-3) - 8/10 examples done
  - ✅ basic_evaluation.py
  - ✅ custom_criteria_example.py
  - ✅ pairwise_comparison_example.py
  - ✅ multiple_evaluators.py
  - ✅ middleware_usage.py
  - ✅ error_handling_example.py
  - ✅ provider_switching.py
  - ✅ evaluator_registry_example.py
- ✅ Test Coverage Expansion (Week 2-3) - ~165+ tests across 8 files
  - ✅ test_semantic.py (25 tests)
  - ✅ test_api.py (24 tests)
  - ✅ test_base.py (22 tests)
  - ✅ test_models.py (40+ tests)
  - ✅ test_custom_criteria.py (18 tests)
  - ✅ test_pairwise.py (25 tests)
  - ✅ test_error_handling.py (12 tests)
  - ✅ test_registry.py (20+ tests)
- ✅ Evaluator Registry & Validation (Week 2-3) - Registry system with custom evaluator support
- ✅ Test Infrastructure Refactoring (Nov 12) - Eliminated ~158 lines of duplicate code

**Remaining (20%):**
- ⏳ User guides (quickstart, installation, custom evaluators)
- ⏳ Troubleshooting guide
- ⏳ Optional examples (RAG evaluation, production setup)

**Blocked:**
- None

**Bonus Work Completed:**
- ✅ Test infrastructure refactoring (conftest.py, eliminated duplication)
- ✅ Registry system (significantly improves extensibility)
- ✅ Documentation generation (test-improvements.md, multiple-evaluators.md, evaluator-registry.md)

**Next Up:**
- ✅ Batch example (batch_manual.py) - COMPLETED
- ✅ Advanced config example (advanced_config.py) - COMPLETED
- ✅ API documentation (MkDocs setup + 16 API pages) - COMPLETED
- ⏳ User guides (quickstart exists, need installation, troubleshooting)
- ⏳ Optional examples (RAG evaluation, production setup)

---

## Daily Standup Notes

### 2025-11-12 (Tuesday)
**Done Yesterday:**
- Completed comprehensive Phase 2 review
- Created 5-document context architecture plan
- Created DESIGN_SPEC.md, AGENTS.md, PROJECT_PLAN.md, PROJECT_TODO.md

**Doing Today:**
- Updating README.md
- Starting CustomCriteriaEvaluator planning

**Blockers:**
- None

---

## Decisions Made During Phase 2.5

| Date | Decision | Rationale | Impact |
|------|----------|-----------|--------|
| 2025-11-22 | CustomCriteria before Milvus | Higher ROI, unblocks users | Delays Phase 3 |
| 2025-11-22 | compare() separate from evaluate() | Different API contract | Cleaner API design |
| TBD | ... | ... | ... |

---

## Notes & Learnings

### What's Working Well
- Template method pattern makes new evaluators easy
- PydanticAI providing type safety
- Automatic interaction tracking is unique

### Challenges
- Documentation takes longer than expected
- Need more real-world testing
- Examples need to cover edge cases

### Ideas for Future
- Web UI for result exploration (Phase 10+)
- Integration with experiment tracking tools
- Community evaluator marketplace

---

## Blockers & Risks

### Current Blockers
- None

### Potential Risks
1. **Scope Creep**
   - Risk: Adding too many features to Phase 2.5
   - Mitigation: Stick to must-haves only

2. **Documentation Time**
   - Risk: 10+ examples takes longer than 5-7 days
   - Mitigation: Start simple, iterate

3. **API Design**
   - Risk: compare() API might need iteration
   - Mitigation: Get feedback early

---

## Quick Reference

### Commands
```bash
# Run tests
make test

# Type check
make type-check

# Lint
make lint

# Format
make format

# All checks
make test && make type-check && make lint
```

### File Locations
- Evaluators: `arbiter/evaluators/`
- Examples: `examples/`
- Tests: `tests/unit/` and `tests/integration/`
- Docs: This file + DESIGN_SPEC.md + AGENTS.md

### Who to Ask
- Architecture questions: Read DESIGN_SPEC.md
- Implementation patterns: Read AGENTS.md, look at evaluators/semantic.py
- Roadmap questions: Read PROJECT_PLAN.md

---

## Next Session Checklist

**Before Starting Next Session:**
- [ ] Read this file (PROJECT_TODO.md)
- [ ] Check git status
- [ ] Review recent commits
- [ ] Check for new issues/PRs

**When Starting Work:**
- [ ] Update "Daily Standup Notes" section
- [ ] Check off completed tasks
- [ ] Update "Progress Tracking" percentage
- [ ] Create feature branch if needed

**When Ending Session:**
- [ ] Update this file with progress
- [ ] Check in code if stable
- [ ] Note any blockers
- [ ] Update "Next Up" section

---

## Related Documents

**Essential Context (4-Layer System):**
1. **Global Context:** ~/.claude/CLAUDE.md (universal rules)
2. **Project Context:** AGENTS.md (how to work here)
3. **Running Context:** THIS FILE (current milestone)
4. **Prompt Context:** Your immediate request

**Reference Documents:**
- **DESIGN_SPEC.md** - Vision and architecture
- **PROJECT_PLAN.md** - Complete multi-milestone roadmap
- **PHASE2_REVIEW.md** - Phase 2 assessment
- **EVALUATOR_RECOMMENDATIONS.md** - Evaluator priorities
- **CONTRIBUTING.md** - Development workflow

---

**Last Updated:** 2025-11-12 | **Next Review:** 2025-11-19 (weekly)
