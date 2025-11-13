# PROJECT_TODO - Current Milestone Tracker

**Type:** Running Context (Layer 3 of 4-layer context framework)
**Current Milestone:** Phase 2.5 - Fill Critical Gaps
**Duration:** 2-3 weeks (Nov 22 - Dec 12, 2025)
**Status:** 🚧 IN PROGRESS (0% complete)
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

### 🔴 Priority 1: CustomCriteriaEvaluator Implementation

**Estimated Time:** 2-3 days
**Status:** ⏳ NOT STARTED

#### Tasks

- [ ] **Day 1: Single Criteria Mode**
  - [ ] Create `evaluators/custom_criteria.py`
  - [ ] Define `CustomCriteriaResponse` model
    - score: float (0-1)
    - confidence: float (default 0.85)
    - explanation: str
    - criteria_met: List[str]
    - criteria_not_met: List[str]
  - [ ] Implement `CustomCriteriaEvaluator` class
    - [ ] `_get_system_prompt()` - Expert evaluator role
    - [ ] `_get_user_prompt()` - Format criteria and output
    - [ ] `_get_response_type()` - Return CustomCriteriaResponse
    - [ ] `_compute_score()` - Extract Score from response
  - [ ] Test single criteria mode
  - [ ] Export in __init__.py files

- [ ] **Day 2: Multi-Criteria Mode**
  - [ ] Add multi-criteria support (dict input)
  - [ ] Return multiple Score objects (one per criterion)
  - [ ] Update response model for multi-criteria
  - [ ] Test multi-criteria mode

- [ ] **Day 3: Testing & Documentation**
  - [ ] Write unit tests
    - [ ] Single criteria evaluation
    - [ ] Multi-criteria evaluation
    - [ ] Edge cases (empty criteria, long criteria)
    - [ ] Error handling
  - [ ] Achieve >80% test coverage
  - [ ] Write docstrings
  - [ ] Create example (examples/custom_criteria_example.py)
  - [ ] Update main __init__.py exports

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

**Deliverables:**
- [ ] CustomCriteriaEvaluator class working
- [ ] Single and multi-criteria modes
- [ ] Tests (>80% coverage)
- [ ] Example code
- [ ] Documentation

---

## Week 2: PairwiseComparisonEvaluator (Nov 29 - Dec 5)

### 🔴 Priority 2: PairwiseComparisonEvaluator Implementation

**Estimated Time:** 2-3 days
**Status:** ⏳ NOT STARTED

#### Tasks

- [ ] **Day 1: Core Comparison Logic**
  - [ ] Create `evaluators/pairwise.py`
  - [ ] Define `ComparisonResult` model (not EvaluationResult)
    - winner: Literal["output_a", "output_b", "tie"]
    - confidence: float
    - reasoning: str
    - aspect_scores: Dict[str, Dict[str, float]]
  - [ ] Define `PairwiseResponse` model
    - winner: str
    - confidence: float
    - reasoning: str
    - aspect_comparisons: List[AspectComparison]
  - [ ] Implement `PairwiseComparisonEvaluator`
    - [ ] Handle two outputs instead of one
    - [ ] Aspect-level comparison
    - [ ] Winner determination logic

- [ ] **Day 2: API Design**
  - [ ] Create `compare()` function in api.py
    - Different from evaluate() - takes output_a, output_b
    - Returns ComparisonResult (not EvaluationResult)
    - Handles criteria string or list
  - [ ] Test comparison API
  - [ ] Handle tie cases
  - [ ] Confidence scoring

- [ ] **Day 3: Testing & Documentation**
  - [ ] Write unit tests
    - [ ] Basic comparison (A wins)
    - [ ] Basic comparison (B wins)
    - [ ] Tie cases
    - [ ] Aspect-level comparison
    - [ ] Multiple criteria
    - [ ] Error handling
  - [ ] Achieve >80% test coverage
  - [ ] Write docstrings
  - [ ] Create example (examples/pairwise_comparison_example.py)
  - [ ] Update __init__.py exports

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
- [ ] PairwiseComparisonEvaluator class
- [ ] compare() API function
- [ ] ComparisonResult model
- [ ] Aspect-level comparison
- [ ] Tests (>80% coverage)
- [ ] Example code
- [ ] Documentation

---

## Week 2-3: Quality & Documentation (Dec 2-12)

### 🟡 Priority 3: Multi-Evaluator Error Handling

**Estimated Time:** 1 day
**Status:** ⏳ NOT STARTED

#### Tasks

- [ ] **Error Handling Improvements**
  - [ ] Add `errors` field to EvaluationResult
  - [ ] Add `partial` flag to EvaluationResult
  - [ ] Graceful degradation when one evaluator fails
  - [ ] Clear error messages in result
  - [ ] Tests for partial failure scenarios
  - [ ] Documentation

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
- [ ] Partial result support
- [ ] Error tracking
- [ ] Tests
- [ ] Documentation

---

### 🟡 Priority 4: Evaluator Registry & Validation

**Estimated Time:** 1 day
**Status:** ⏳ NOT STARTED

#### Tasks

- [ ] **Registry System**
  - [ ] Create AVAILABLE_EVALUATORS dict
    - Maps name to evaluator class
    - Allows registration of custom evaluators
  - [ ] Add validation in evaluate()
    - Check evaluator name is valid
    - Raise ValidationError with helpful message
  - [ ] Add type hints (Literal)
    - EvaluatorName = Literal["semantic", "custom_criteria", ...]
    - IDE autocomplete support
  - [ ] Tests for validation
  - [ ] Documentation

**Example:**
```python
# Good
result = await evaluate(evaluators=["semantic"])  # Works

# Bad - clear error
result = await evaluate(evaluators=["unknown"])
# Raises: ValidationError("Unknown evaluator: 'unknown'. Available: ['semantic', 'custom_criteria', ...]")
```

**Deliverables:**
- [ ] Evaluator registry
- [ ] Validation logic
- [ ] Type hints
- [ ] Tests
- [ ] Documentation

---

### 🟢 Priority 5: Documentation & Examples

**Estimated Time:** 5-7 days
**Status:** ⏳ NOT STARTED

#### Tasks

- [ ] **Examples (10-15 files)**
  - [ ] examples/1_basic_evaluation.py - Simple semantic evaluation
  - [ ] examples/2_custom_criteria.py - Domain-specific criteria
  - [ ] examples/3_pairwise_comparison.py - A/B testing
  - [ ] examples/4_multiple_evaluators.py - Combining evaluators
  - [ ] examples/5_middleware_usage.py - Logging, metrics, caching
  - [ ] examples/6_cost_tracking.py - Token usage and cost analysis
  - [ ] examples/7_error_handling.py - Handling failures gracefully
  - [ ] examples/8_batch_manual.py - Manual batching with asyncio.gather
  - [ ] examples/9_provider_switching.py - Using different providers
  - [ ] examples/10_advanced_config.py - Temperature, retries, etc.
  - [ ] examples/11_direct_evaluator.py - Using evaluators directly
  - [ ] examples/12_interaction_tracking.py - Accessing LLM interactions
  - [ ] examples/13_confidence_filtering.py - Filter by confidence
  - [ ] examples/14_rag_evaluation.py - RAG system evaluation pattern
  - [ ] examples/15_production_setup.py - Production best practices

- [ ] **API Reference Documentation**
  - [ ] Document evaluate() function
  - [ ] Document compare() function
  - [ ] Document all evaluators
  - [ ] Document EvaluationResult model
  - [ ] Document ComparisonResult model
  - [ ] Document middleware
  - [ ] Document LLMManager

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

### Overall Phase 2.5 Progress: 0%

**Completed:**
- None yet (just starting)

**In Progress:**
- Documentation planning

**Blocked:**
- None

**Next Up:**
- CustomCriteriaEvaluator implementation

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
