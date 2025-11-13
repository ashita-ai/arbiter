# Arbiter Design Specification

**Version:** 1.0
**Date:** November 12, 2025
**Status:** Phase 2 Complete, Phase 2.5 Starting

---

## Executive Summary

**Arbiter** is a production-grade LLM evaluation framework providing simple APIs, complete observability, and provider-agnostic infrastructure for AI teams at scale.

**Core Value Proposition:** Evaluate LLM outputs with 3 lines of code while maintaining full visibility into cost, quality, and decision-making processes.

**Target Market:** AI engineers, MLOps teams, and product teams deploying LLMs in production who need reliable evaluation without complexity.

---

## 1. Vision & Mission

### Vision
Be the pragmatic choice for production LLM evaluation - simple enough for quick adoption, powerful enough for scale, transparent enough for confidence.

### Mission
Provide production-grade evaluation infrastructure that:
1. Makes evaluation accessible (simple API)
2. Provides complete visibility (observability)
3. Works at any scale (streaming, pooling, middleware)
4. Supports any LLM provider (provider-agnostic)
5. Enables extensibility (plugins, middleware, custom evaluators)

### Strategic Position

**"Simple, production-grade LLM evaluation with complete observability"**

Not an experiment tracking tool, not a prompt engineering playground - infrastructure-grade evaluation for production systems.

---

## 2. Problem Statement

### The Challenge

LLM evaluation is critical for production systems, but existing tools fall short:

#### Problem 1: Complexity
- **LangChain evaluators** require deep framework knowledge
- **Steep learning curves** delay adoption
- **Complex configuration** makes simple tasks hard

#### Problem 2: Poor Observability
- **Black-box evaluation** - unclear why scores are what they are
- **No cost visibility** - token usage and costs hidden
- **Missing audit trails** - can't trace decisions

#### Problem 3: Vendor Lock-In
- **OpenAI Evals** only works with OpenAI
- **Provider-specific tools** create switching costs
- **Future migration risk** as providers evolve

#### Problem 4: Experimental Focus
- **Lack production features** (retry, pooling, rate limiting)
- **Missing streaming/batch** for scale
- **No reliability guarantees** for critical systems

### Market Gap

Teams want:
- Simple API (like OpenAI Evals)
- Production features (like TruLens)
- Provider-agnostic (unlike OpenAI Evals)
- Focused on evaluation (unlike LangChain)
- Open source (unlike Braintrust)

**Arbiter fills this gap.**

---

## 3. Target Users

### Primary: AI Engineers
**Need:** Evaluate LLM outputs quickly and reliably
**Pain:** Complex tools slow down iteration
**Value:** "Evaluate with 3 lines of code, full cost visibility"

**Use Cases:**
- RAG system quality assessment
- Prompt optimization
- Model comparison
- Output validation

### Secondary: MLOps/Platform Teams
**Need:** Production-ready evaluation infrastructure
**Pain:** Experimental tools lack scalability
**Value:** "Production infrastructure with streaming, pooling, middleware"

**Use Cases:**
- Build internal AI platforms
- Integrate evaluation into pipelines
- Monitor production quality
- Cost optimization

### Tertiary: Technical Leadership
**Need:** Confident decision-making on LLM quality
**Pain:** Lack of transparency in evaluation
**Value:** "Make confident decisions with comprehensive audit trails"

**Use Cases:**
- Vendor selection
- Quality benchmarking
- Compliance requirements
- Budget planning

---

## 4. Key Features

### 4.1 Simple API ⭐⭐⭐⭐⭐

**Minimal Example:**
```python
from arbiter import evaluate

result = await evaluate(
    output="Paris is the capital of France",
    reference="The capital of France is Paris",
    evaluators=["semantic"],
    model="gpt-4o-mini"
)

print(f"Score: {result.overall_score:.2f}")
print(f"Passed: {result.passed}")
```

**Characteristics:**
- 3-line usage for common cases
- Automatic client management
- Sensible defaults
- Progressive complexity (simple → advanced)

### 4.2 Complete Observability ⭐⭐⭐⭐⭐

**Every LLM call tracked automatically:**
```python
# Access all interactions
for interaction in result.interactions:
    print(f"Purpose: {interaction.purpose}")
    print(f"Model: {interaction.model}")
    print(f"Tokens: {interaction.tokens_used}")
    print(f"Latency: {interaction.latency}s")
    print(f"Cost: ${interaction.tokens_used * 0.00001:.6f}")
```

**Capabilities:**
- Automatic LLM interaction tracking
- Token usage and cost calculation
- Performance metrics
- Complete audit trails
- Confidence scoring with explanations

**Differentiator:** No other framework provides automatic interaction tracking at the evaluator level.

### 4.3 Provider-Agnostic Design ⭐⭐⭐⭐⭐

**Supports multiple providers:**
- OpenAI (GPT-4o, GPT-4, GPT-3.5-turbo)
- Anthropic (Claude 3.5 Sonnet, Claude 3 Opus)
- Google (Gemini 1.5 Pro/Flash)
- Groq (Llama 3.1, Mixtral)
- Mistral AI
- Cohere

**Easy switching:**
```python
# Use any provider
client = await LLMManager.get_client(
    provider="anthropic",
    model="claude-3-5-sonnet"
)
```

### 4.4 Production Features ⭐⭐⭐⭐⭐

**Built-in production optimizations:**

1. **Connection Pooling**
   - Reduces latency
   - Manages resources efficiently
   - Health checks and auto-cleanup

2. **Automatic Retry**
   - Exponential backoff
   - Configurable strategies (quick, standard, persistent)
   - Selective retry (only transient errors)

3. **Middleware System**
   - Logging
   - Metrics collection
   - Caching (LRU eviction)
   - Rate limiting

4. **Performance Monitoring**
   - Per-operation metrics
   - Token tracking
   - Cost analysis
   - Error tracking

### 4.5 Extensible Architecture ⭐⭐⭐⭐

**Plugin System:**
- Custom evaluators (implement 4 methods)
- Custom middleware (implement 1 method)
- Custom storage backends (implement protocol)
- Custom providers (extend enum)

**Template Method Pattern:**
```python
class MyEvaluator(BasePydanticEvaluator):
    def _get_system_prompt(self) -> str:
        return "Your evaluation instructions..."

    def _get_user_prompt(self, output, reference, criteria) -> str:
        return f"Evaluate: {output}"

    def _get_response_type(self):
        return MyResponse

    async def _compute_score(self, response) -> Score:
        return Score(name=self.name, value=response.score)
```

---

## 5. Architecture

### 5.1 Layered Design

```
┌─────────────────────────────────────┐
│       Public API (api.py)           │  Simple entry point
│  evaluate(), compare()              │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│      Middleware Pipeline            │  Cross-cutting concerns
│  Logging, Metrics, Caching,         │
│  Rate Limiting                      │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│   Evaluators (via LLM Clients)      │  Business logic
│  ├─ SemanticEvaluator              │
│  ├─ FactualityEvaluator (planned)   │
│  └─ Custom Evaluators               │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│   LLM Client + Connection Pool      │  Provider abstraction
│  (OpenAI, Anthropic, Google, etc.)  │
├─ PydanticAI Integration            │
├─ Automatic Retry Logic             │
└─ Token Tracking                    │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│     External LLM APIs               │  External services
│  (OpenAI, Anthropic, Gemini, etc.)  │
└─────────────────────────────────────┘
```

### 5.2 Core Principles

#### SOLID Principles
- **Single Responsibility:** Each class has one reason to change
- **Open/Closed:** Extensible via evaluator interface
- **Liskov Substitution:** Evaluators are substitutable
- **Interface Segregation:** Small, focused protocols
- **Dependency Inversion:** Depends on abstractions (BaseEvaluator, StorageBackend)

#### Design Patterns
- **Template Method:** BasePydanticEvaluator forces consistent structure
- **Chain of Responsibility:** Middleware pipeline
- **Object Pool:** LLMClientPool manages connections
- **Strategy:** Multiple evaluators implement common interface
- **Singleton:** LLMManager static methods

#### Type Safety
- **Strict mypy:** All functions typed, no `Any` without reason
- **Pydantic models:** Runtime validation throughout
- **PydanticAI:** Structured LLM outputs
- **Protocol-based:** Duck typing with type safety

### 5.3 Data Models

#### Core Models (Pydantic)

**Score** - Individual evaluation metric:
```python
- name: str              # "semantic_similarity"
- value: float (0-1)    # Normalized score
- confidence: float     # Evaluator confidence
- explanation: str      # Human-readable reasoning
- metadata: Dict        # Additional context
```

**LLMInteraction** - Complete LLM call record:
```python
- prompt: str           # Sent to LLM
- response: str         # LLM's response
- model: str            # Model used
- tokens_used: int      # Token count
- latency: float        # Response time
- timestamp: datetime   # When called
- purpose: str          # Evaluation purpose
- metadata: Dict        # Evaluator info
```

**EvaluationResult** - Complete evaluation outcome:
```python
# Inputs
- output: str
- reference: Optional[str]
- criteria: Optional[str]

# Results
- scores: List[Score]
- overall_score: float (0-1)
- passed: bool (threshold comparison)

# Metadata
- metrics: List[Metric]
- evaluator_names: List[str]
- total_tokens: int
- processing_time: float
- interactions: List[LLMInteraction]  # COMPLETE AUDIT TRAIL
```

---

## 6. Differentiation

### vs. LangChain Evaluators
**Their Strength:** Large ecosystem, many evaluators
**Their Weakness:** Complex API, requires LangChain framework
**Arbiter Advantage:** Simpler API, better observability, framework-agnostic

### vs. TruLens
**Their Strength:** Excellent observability, production-ready
**Their Weakness:** Opinionated about RAG, heavier framework
**Arbiter Advantage:** More general-purpose, lighter weight, simpler API

### vs. DeepEval
**Their Strength:** pytest integration, many built-in metrics
**Their Weakness:** Testing-focused, no streaming
**Arbiter Advantage:** Production-focused, streaming support (planned)

### vs. Ragas
**Their Strength:** Best-in-class RAG evaluation metrics
**Their Weakness:** RAG-only focus
**Arbiter Advantage:** General-purpose, broader applicability

### vs. OpenAI Evals
**Their Strength:** Simple, OpenAI-maintained
**Their Weakness:** OpenAI-only, limited features
**Arbiter Advantage:** Provider-agnostic, production features

### vs. Braintrust
**Their Strength:** Full platform with UI, dataset management
**Their Weakness:** Commercial, opinionated workflows
**Arbiter Advantage:** Open source, infrastructure-focused, flexible

### Competitive Matrix

| Feature | Arbiter | LangChain | TruLens | DeepEval | Ragas | OpenAI |
|---------|---------|-----------|---------|----------|-------|--------|
| **API Simplicity** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Observability** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| **Provider-Agnostic** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐ |
| **Production** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐ |
| **Type Safety** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |

**Key Differentiators:**
1. Automatic LLM interaction tracking (unique)
2. Simple API + production features (rare combination)
3. Provider-agnostic from ground up
4. Open source infrastructure focus

---

## 7. Success Metrics

### Technical Metrics
- ⭐ **8+ evaluators available** (competitive with market)
- ⭐ **>80% test coverage** (quality assurance)
- ⭐ **<200ms p95 latency** (single evaluator, production performance)
- ⭐ **>0.85 agreement with human judgment** (evaluator quality)

### Adoption Metrics
- ⭐ **100+ GitHub stars** (Month 3)
- ⭐ **10+ production users** (Month 6)
- ⭐ **5+ community evaluators** (Month 9)
- ⭐ **Integration with 2+ major frameworks** (Month 6 - LangChain, LlamaIndex)

### Quality Metrics
- ⭐ **Clear documentation** for all evaluators
- ⭐ **Performance benchmarks** published
- ⭐ **Calibration data** available (evaluator reliability)
- ⭐ **Active community** engagement

### Business Metrics
- **Evaluation Cost Reduction:** 30-50% via caching and cheaper models
- **Time to First Evaluation:** <10 minutes from install
- **User Satisfaction:** >4.5/5 in surveys
- **Retention:** >70% monthly active users return

---

## 8. Roadmap Overview

### Phase 1: Foundation ✅ COMPLETE
**Goal:** Project setup and core infrastructure
**Duration:** 2 weeks
**Status:** Done

### Phase 2: Core Evaluation Engine ✅ COMPLETE
**Goal:** Working evaluation with SemanticEvaluator
**Duration:** 1 week
**Status:** Done + critical fixes applied

### Phase 2.5: Fill Critical Gaps 🚧 CURRENT
**Goal:** CustomCriteria, PairwiseComparison, docs
**Duration:** 2-3 weeks
**Status:** Starting now

**Tasks:**
1. CustomCriteriaEvaluator (2-3 days) 🔴 CRITICAL
2. PairwiseComparisonEvaluator (2-3 days) 🔴 CRITICAL
3. Multi-evaluator error handling (1 day)
4. Documentation + 10 examples (5-7 days)

### Phase 3: Semantic Comparison ⏳ NEXT
**Goal:** Milvus integration for vector similarity
**Duration:** 2 weeks
**Status:** Planned

### Phase 4: Storage & Batch ⏳ PLANNED
**Goal:** Storage backends and batch operations
**Duration:** 2 weeks
**Status:** Planned

### Phase 5: Core Evaluators ⏳ PLANNED
**Goal:** 6-8 production evaluators
**Duration:** 6 weeks
**Evaluators:** Factuality, Relevance, Toxicity, Groundedness, Consistency, ContextRelevance

### Phase 6: Quality Assurance ⏳ PLANNED
**Goal:** Evaluator validation and calibration tools
**Duration:** 2-3 weeks
**Differentiator:** Only framework with built-in quality assurance

### Phase 7-9: Polish & Release ⏳ FUTURE
**Goal:** CI/CD, PyPI package, documentation site
**Duration:** 4 weeks
**Status:** Future

**Total Timeline:** 7 months to v1.0

---

## 9. Technical Decisions

### 9.1 PydanticAI as Foundation
**Decision:** Use PydanticAI instead of raw LLM APIs
**Rationale:** Structured outputs, type safety, provider abstraction
**Trade-off:** Dependency on relatively new library
**Mitigation:** Abstract PydanticAI behind internal interfaces

### 9.2 Template Method for Evaluators
**Decision:** Force evaluators to implement 4 methods
**Rationale:** Consistency, automatic tracking, reduced boilerplate
**Trade-off:** Less flexibility for unusual evaluators
**Mitigation:** Allow bypass for advanced users

### 9.3 Automatic Interaction Tracking
**Decision:** Track every LLM call in base evaluator
**Rationale:** Complete observability without manual instrumentation
**Trade-off:** Slight performance overhead
**Mitigation:** Negligible overhead, huge value

### 9.4 Provider-Agnostic from Day 1
**Decision:** Support multiple providers from start
**Rationale:** Future-proofs user investments
**Trade-off:** More complex than OpenAI-only
**Mitigation:** LLMClient abstraction handles complexity

### 9.5 LLM-as-Judge Pattern
**Decision:** Use LLMs for evaluation (not just metrics)
**Rationale:** More nuanced, semantic understanding
**Trade-off:** Cost and latency
**Mitigation:** Caching, cheaper models, batch operations

---

## 10. Constraints & Non-Goals

### Constraints
1. **Provider-Agnostic:** Must work with any LLM provider
2. **Type Safety:** Strict mypy, Pydantic throughout
3. **Production-Grade:** Retry, pooling, monitoring required
4. **Simple API:** 3-line usage for common cases
5. **Open Source:** MIT license, community-driven

### Non-Goals
1. **Not a UI:** Focus on infrastructure, not dashboards
2. **Not experiment tracking:** Leave to W&B, MLflow
3. **Not prompt engineering:** Focus on evaluation
4. **Not training:** Evaluation only, not model training
5. **Not data labeling:** Evaluation, not annotation

### What We Won't Build
- Web UI for result exploration (use Jupyter, custom dashboards)
- Experiment tracking platform (integrate with existing tools)
- Prompt optimization engine (focus on evaluation)
- Dataset management (users bring their own data)

---

## 11. Dependencies & Tech Stack

### Core Dependencies
- **Python:** 3.10+ (modern features)
- **Pydantic:** 2.12+ (validation)
- **PydanticAI:** 1.14+ (structured LLM outputs)
- **Pymilvus:** 2.6+ (vector database)
- **HTTPX:** 0.28+ (async HTTP)
- **OpenAI SDK:** 2.0+ (LLM provider)

### Optional Dependencies
- **Provider SDKs:** Anthropic, Google, Mistral, Cohere
- **ByteWax:** 0.20+ (streaming)
- **Redis:** 5.0+ (storage backend)

### Development Tools
- **pytest:** Testing framework
- **black:** Code formatting
- **ruff:** Linting
- **mypy:** Type checking (strict mode)

---

## 12. Risk Management

### Technical Risks
1. **PydanticAI Breaking Changes**
   - Risk: Relatively new library
   - Mitigation: Abstract behind interfaces

2. **LLM-as-Judge Reliability**
   - Risk: Inconsistent, expensive
   - Mitigation: Calibration tools, hybrid approaches

3. **Cost at Scale**
   - Risk: Expensive with GPT-4
   - Mitigation: Caching, cheaper models, batch ops

### Market Risks
1. **Competition from Established Players**
   - Risk: LangChain has network effects
   - Mitigation: Focus on production, integrations

2. **Provider Native Solutions**
   - Risk: OpenAI/Anthropic might build in
   - Mitigation: Provider-agnostic, workflow focus

### Adoption Risks
1. **Evaluator Coverage Gap** 🔴 HIGHEST
   - Risk: Only semantic evaluator currently
   - Mitigation: Accelerate Phase 5, CustomCriteria first

2. **Documentation Gap**
   - Risk: Users can't onboard
   - Mitigation: Invest in docs early

---

## 13. Appendix

### Related Documents
- **PROJECT_PLAN.md** - Detailed multi-milestone roadmap
- **PROJECT_TODO.md** - Current milestone tracker
- **AGENTS.md** - How to work with this repository
- **PHASE2_REVIEW.md** - Comprehensive Phase 2 assessment
- **EVALUATOR_RECOMMENDATIONS.md** - Evaluator priorities

### References
- [PydanticAI Documentation](https://ai.pydantic.dev/)
- [Milvus Documentation](https://milvus.io/docs)
- [LangChain Evaluators](https://python.langchain.com/docs/guides/evaluation/)
- [TruLens](https://www.trulens.org/)
- [OpenAI Evals](https://github.com/openai/evals)

### Version History
- **v1.0** (2025-11-12) - Initial design specification

---

**End of Design Specification**
