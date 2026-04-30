"""
Agent - AI agent framework for Nono.

Provides the Nono Agent Architecture (NAA) — a multi-agent system built on
top of Nono's unified connector layer.  Fully standalone — no external
agent-framework dependency required.

Core components:
    - **BaseAgent**: Abstract base class for all agents.
    - **Agent / LlmAgent**: LLM-powered agent with tool calling.
    - **SequentialAgent**: Run sub-agents one after another.
    - **ParallelAgent**: Run sub-agents concurrently.
    - **LoopAgent**: Repeat sub-agents until a condition is met.
    - **MapReduceAgent**: Fan-out to mappers, reduce into one result.
    - **ConsensusAgent**: Multiple agents vote, a judge synthesises.
    - **ProducerReviewerAgent**: Iterative produce-then-review loop.
    - **RouterAgent**: LLM-based dynamic routing to sub-agents.
    - **DebateAgent**: Adversarial debate with two agents and a judge.
    - **EscalationAgent**: Try agents in order, stop at first success.
    - **SupervisorAgent**: LLM-powered supervisor that delegates and evaluates.
    - **VotingAgent**: Majority-vote orchestration without LLM judge.
    - **HandoffAgent**: Peer-to-peer handoff with context transfer.
    - **GroupChatAgent**: N-agent group chat with speaker selection.
    - **HierarchicalAgent**: Multi-level tree orchestration with LLM manager.
    - **GuardrailAgent**: Pre/post-validation wrapper with retry.
    - **BestOfNAgent**: Same agent N times, pick best via scoring.
    - **BatchAgent**: Process item list through one agent with concurrency.
    - **CascadeAgent**: Progressive stages with confidence gates.
    - **TreeOfThoughtsAgent**: Branching reasoning with evaluate + prune.
    - **PlannerAgent**: LLM decomposes task, executes respecting dependencies.
    - **SubQuestionAgent**: Decompose complex question, dispatch, synthesise.
    - **ContextFilterAgent**: Filter context/history per sub-agent.
    - **ReflexionAgent**: Iterative self-improvement with persistent memory.
    - **SpeculativeAgent**: Race multiple agents, cancel losers early.
    - **CircuitBreakerAgent**: Protect against cascading agent failures.
    - **TournamentAgent**: Pairwise bracket elimination with LLM judge.
    - **ShadowAgent**: Dark-launch candidate alongside stable agent.
    - **CompilerAgent**: Meta-optimise sub-agent instructions against a dataset.
    - **CheckpointableAgent**: Pause, serialise state, and resume a pipeline.
    - **DynamicFanOutAgent**: LLM-determined parallel fan-out and reduce.
    - **SwarmAgent**: Fluid agent handoffs with evolving context variables.
    - **MemoryConsolidationAgent**: Compress history when context grows long.
    - **PriorityQueueAgent**: Execute sub-agents by priority order.
    - **MonteCarloAgent**: MCTS tree search with UCT exploration.
    - **GraphOfThoughtsAgent**: DAG-based thought generation, aggregation and scoring.
    - **BlackboardAgent**: Shared blackboard architecture with expert activation.
    - **MixtureOfExpertsAgent**: Gating + weighted multi-expert combination.
    - **CoVeAgent**: Chain-of-Verification anti-hallucination pipeline.
    - **SagaAgent**: Distributed transactions with compensating rollback.
    - **LoadBalancerAgent**: Distribute requests across equivalent agents.
    - **EnsembleAgent**: Aggregate outputs from multiple agents.
    - **TimeoutAgent**: Deadline wrapper with fallback.
    - **AdaptivePlannerAgent**: Re-plan after every step based on results.
    - **SkeletonOfThoughtAgent**: Parallel elaboration of outline points.
    - **LeastToMostAgent**: Solve from easiest to hardest sub-problems.
    - **SelfDiscoverAgent**: LLMs self-compose reasoning structures.
    - **GeneticAlgorithmAgent**: Evolutionary optimisation with crossover.
    - **MultiArmedBanditAgent**: Online learning router (explore/exploit).
    - **SocraticAgent**: Iterative questioning to deepen understanding.
    - **MetaOrchestratorAgent**: Chooses orchestration pattern, not agent.
    - **CacheAgent**: Semantic caching to avoid redundant LLM calls.
    - **BudgetAgent**: Cost-aware wrapper with token budget caps.
    - **CurriculumAgent**: Progressive task generation with skill library.
    - **SelfConsistencyAgent**: Majority-vote over N samples of the same agent.
    - **MixtureOfAgentsAgent**: Multi-layer proposers + aggregators pipeline.
    - **StepBackAgent**: Abstract first, then reason with the abstraction.
    - **OrchestratorWorkerAgent**: Iterative plan → delegate → evaluate loop.
    - **SelfRefineAgent**: Generate → self-critique → refine iteration.
    - **BacktrackingAgent**: Pipeline with validation and rewind on failure.
    - **ChainOfDensityAgent**: Iterative densification of content.
    - **MediatorAgent**: Neutral compromise synthesis from competing proposals.
    - **DivideAndConquerAgent**: Recursive split → solve → merge.
    - **BeamSearchAgent**: K beams with expand → score → prune per step.
    - **RephraseAndRespondAgent**: Rephrase query, then solve with richer framing.
    - **CumulativeReasoningAgent**: Proposer → verifier → reporter incremental facts.
    - **MultiPersonaAgent**: Single agent adopts multiple personas, then synthesises.
    - **AntColonyAgent**: Pheromone-guided parallel exploration with evaporation.
    - **PipelineParallelAgent**: Assembly-line stages processing item lists.
    - **ContractNetAgent**: Competitive bidding, highest bidder executes.
    - **RedTeamAgent**: Adversarial attacker-defender hardening loop.
    - **FeedbackLoopAgent**: Circular chain with convergence check.
    - **WinnowingAgent**: Progressive candidate elimination by global ranking.
    - **MixtureOfThoughtsAgent**: Parallel reasoning strategies fused by selector.
    - **SimulatedAnnealingAgent**: Temperature-decaying acceptance of worse solutions.
    - **TabuSearchAgent**: Local search with memory to prevent cycling.
    - **ParticleSwarmAgent**: Personal-best + global-best swarm guidance.
    - **DifferentialEvolutionAgent**: Mutate by differencing population members.
    - **BayesianOptimizationAgent**: Surrogate model + acquisition function search.
    - **AnalogicalReasoningAgent**: Self-generate analogies, then solve.
    - **ThreadOfThoughtAgent**: Segment-by-segment context walk-through.
    - **ExpertPromptingAgent**: Auto-generate expert identity, then answer.
    - **BufferOfThoughtsAgent**: Distil and reuse thought-templates.
    - **ChainOfAbstractionAgent**: Reason with placeholders, then ground.
    - **VerifierAgent**: Generate N solutions, score each, pick best.
    - **ProgOfThoughtAgent**: Generate Python code, execute, return output.
    - **InnerMonologueAgent**: Closed-loop verbal reasoning with feedback.
    - **RolePlayingAgent**: Two agents converse in assigned roles (CAMEL).
    - **GossipProtocolAgent**: Epidemic information spreading among peers.
    - **AuctionAgent**: Agents bid on tasks, highest bidder executes.
    - **DelphiMethodAgent**: Anonymous iterative expert consensus.
    - **NominalGroupAgent**: Structured idea generation + ranking.
    - **ActiveRetrievalAgent**: Retrieve only when confidence is low (FLARE).
    - **IterativeRetrievalAgent**: Interleave CoT and retrieval (IRCoT).
    - **PromptChainAgent**: Explicit multi-step prompt pipeline.
    - **HypothesisTestingAgent**: Generate, test, falsify, refine hypotheses.
    - **SkillLibraryAgent**: Accumulate and retrieve reusable skills (Voyager).
    - **RecursiveCriticAgent**: Nested multi-depth critique and revision.
    - **DemonstrateSearchPredictAgent**: DSP 3-stage pipeline.
    - **DoubleLoopLearningAgent**: Question assumptions, not just outcomes.
    - **AgendaAgent**: Priority-queue sub-goal planning and resolution.
    - **Runner**: Execute agents with automatic session management.
    - **Session**: Conversation thread with events and state.
    - **Event**: Immutable record of agent actions.
    - **FunctionTool / @tool**: Tool system for LLM function calling.
    - **AgentFactory**: Dynamic agent generation from natural language descriptions.
    - **AgentBlueprint**: Immutable, reviewable spec before instantiation.
    - **OrchestrationBlueprint**: Multi-agent pipeline spec with pattern selection.
    - **OrchestrationSelector**: Recommends orchestration patterns from task descriptions.
    - **OrchestrationRegistry**: Extensible registry for orchestration patterns.
    - **PatternRegistration**: Immutable spec for a registered pattern.
    - **register_pattern()**: Convenience function to register custom patterns.
    - **create_agent_from_prompt()**: One-liner convenience function.
    - **Workspace**: Declarative I/O description for agents.
"""

from .base import (
    BaseAgent,
    ContentItem,
    Event,
    EventType,
    InvocationContext,
    MAX_TRANSFER_DEPTH,
    Session,
    SharedContent,
    AfterAgentCallback,
    AfterToolCallback,
    BeforeAgentCallback,
    BeforeToolCallback,
    # Orchestration lifecycle callbacks
    AgentExecutedCallback,
    AgentExecutingCallback,
    BetweenAgentsCallback,
    OrchestrationEndCallback,
    OrchestrationStartCallback,
)
from .tool import (
    FunctionTool,
    ToolContext,
    ToolIssue,
    tool,
    validate_tools,
)
from .llm_agent import (
    Agent,
    LlmAgent,
    estimate_tokens,
)
from .compaction import (
    CallableStrategy,
    CompactionResult,
    CompactionStrategy,
    SummarizationStrategy,
    TokenAwareStrategy,
)
from nono.connector.connector_genai import StreamChunk
from .execution import (
    # Task Packets
    TaskPacket,
    EscalationPolicy,
    ReportingContract,
    # Worker State Machine
    WorkerState,
    WorkerStateMachine,
    WorkerTransition,
    InvalidTransitionError,
    # Failure Taxonomy
    FailureCategory,
    FailureClassifier,
    ClassifiedFailure,
    RecoveryRecipe,
    # Policy Engine
    PolicyRule,
    PolicyResult,
    PolicyEngine,
    CallablePolicy,
    AutoMergePolicy,
    StaleBranchPolicy,
    StartupRecoveryPolicy,
    LaneCompletionPolicy,
    DegradedModePolicy,
    # Verification Contract
    VerificationLevel,
    VerificationContract,
    VerificationResult,
    # Worktree Isolation
    WorktreeManager,
    WorktreeInfo,
    WorktreeError,
    WorkspaceMismatchError,
    # Stale-Branch Detection
    StaleBranchDetector,
    BranchStatus,
    # Conversation Checkpoints
    ConversationCheckpoint,
    ConversationCheckpointManager,
    # Plan Mode
    PlanModeAgent,
    PlanResult,
)
from .agent_factory import (
    AgentBlueprint,
    AgentFactory,
    BlueprintValidationError,
    DynamicCreationDisabledError,
    OrchestrationBlueprint,
    OrchestrationFactory,
    OrchestrationRegistry,
    OrchestrationSelector,
    PatternRegistration,
    SystemPromptGenerator,
    ToolSelector,
    AgentConfigurator,
    create_agent_from_prompt,
    register_pattern,
)
from .workflow_agents import (
    ActiveRetrievalAgent,
    AdaptivePlannerAgent,
    AgendaAgent,
    AnalogicalReasoningAgent,
    AntColonyAgent,
    AuctionAgent,
    BacktrackingAgent,
    BatchAgent,
    BayesianOptimizationAgent,
    BeamSearchAgent,
    BestOfNAgent,
    BlackboardAgent,
    BudgetAgent,
    BufferOfThoughtsAgent,
    CacheAgent,
    CascadeAgent,
    ChainOfAbstractionAgent,
    ChainOfDensityAgent,
    CheckpointableAgent,
    CircuitBreakerAgent,
    ContractNetAgent,
    CoVeAgent,
    CompilerAgent,
    ConsensusAgent,
    ContextFilterAgent,
    CumulativeReasoningAgent,
    CurriculumAgent,
    DebateAgent,
    DelphiMethodAgent,
    DemonstrateSearchPredictAgent,
    DifferentialEvolutionAgent,
    DivideAndConquerAgent,
    DoubleLoopLearningAgent,
    DynamicFanOutAgent,
    EnsembleAgent,
    EscalationAgent,
    ExpertPromptingAgent,
    FeedbackLoopAgent,
    GeneticAlgorithmAgent,
    GossipProtocolAgent,
    GraphOfThoughtsAgent,
    GroupChatAgent,
    GuardrailAgent,
    HandoffAgent,
    HierarchicalAgent,
    HypothesisTestingAgent,
    InnerMonologueAgent,
    IterativeRetrievalAgent,
    LeastToMostAgent,
    LoadBalancerAgent,
    LoopAgent,
    MapReduceAgent,
    MediatorAgent,
    MemoryConsolidationAgent,
    MetaOrchestratorAgent,
    MixtureOfAgentsAgent,
    MixtureOfExpertsAgent,
    MixtureOfThoughtsAgent,
    MonteCarloAgent,
    MultiArmedBanditAgent,
    MultiPersonaAgent,
    NominalGroupAgent,
    OrchestratorWorkerAgent,
    ParallelAgent,
    ParticleSwarmAgent,
    PipelineParallelAgent,
    PlannerAgent,
    PriorityQueueAgent,
    ProducerReviewerAgent,
    ProgOfThoughtAgent,
    PromptChainAgent,
    RecursiveCriticAgent,
    RedTeamAgent,
    ReflexionAgent,
    RephraseAndRespondAgent,
    RolePlayingAgent,
    RouterAgent,
    SagaAgent,
    SelfConsistencyAgent,
    SelfDiscoverAgent,
    SelfRefineAgent,
    SequentialAgent,
    ShadowAgent,
    SimulatedAnnealingAgent,
    SkeletonOfThoughtAgent,
    SkillLibraryAgent,
    SocraticAgent,
    SpeculativeAgent,
    StepBackAgent,
    SubQuestionAgent,
    SupervisorAgent,
    SwarmAgent,
    TabuSearchAgent,
    ThreadOfThoughtAgent,
    TimeoutAgent,
    TournamentAgent,
    TreeOfThoughtsAgent,
    VerifierAgent,
    VotingAgent,
    WinnowingAgent,
)
from .human_input import HumanInputAgent
from .runner import Runner
from .tasker_tool import json_task_tool, tasker_tool
from .skill import BaseSkill, SkillDescriptor, SkillRegistry, registry, skill_from_agent
from .skill_loader import MarkdownSkill, load_skill_md, scan_skills_dir
from .workspace import (
    CloudStorageEntry,
    FileEntry,
    InlineEntry,
    IODirection,
    OutputEntry,
    StorageKind,
    TemplateEntry,
    URLEntry,
    Workspace,
    WorkspaceEntry,
)
from .tools import (
    ALL_TOOLS,
    DATETIME_TOOLS,
    TEXT_TOOLS,
    WEB_TOOLS,
    PYTHON_TOOLS,
    SHORTFX_TOOLS,
    SHORTFX_DISCOVERY_TOOLS,
    ShortFxSkill,
    shortfx_mcp_tools,
)
from .tracing import (
    LLMCall,
    TokenUsage,
    ToolRecord,
    Trace,
    TraceCollector,
    TraceStatus,
)
from .keepinmind import (
    FileMemoryStore,
    KeepInMind,
    MemoryEntry,
    MemoryStore,
)
from .agent_card import (
    A2A_PROTOCOL_VERSION,
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentProvider,
    AgentSkill,
    load_agent_card,
    save_agent_card,
    serve_agent_card,
    to_agent_card,
)

# Hooks — re-exported from nono.hooks for convenience
from ..hooks import (
    Hook,
    HookContext,
    HookEvent,
    HookManager,
    HookResult,
    load_hooks_from_file,
)

__all__ = [
    # Base
    "BaseAgent",
    "ContentItem",
    "Event",
    "EventType",
    "InvocationContext",
    "MAX_TRANSFER_DEPTH",
    "Session",
    "SharedContent",
    # Callbacks
    "AfterAgentCallback",
    "AfterToolCallback",
    "BeforeAgentCallback",
    "BeforeToolCallback",
    # Orchestration lifecycle callbacks
    "AgentExecutedCallback",
    "AgentExecutingCallback",
    "BetweenAgentsCallback",
    "OrchestrationEndCallback",
    "OrchestrationStartCallback",
    # Hooks
    "Hook",
    "HookContext",
    "HookEvent",
    "HookManager",
    "HookResult",
    "load_hooks_from_file",
    # Tools
    "FunctionTool",
    "ToolContext",
    "ToolIssue",
    "tool",
    "validate_tools",
    # LLM Agent
    "Agent",
    "LlmAgent",
    "estimate_tokens",
    # Compaction
    "CallableStrategy",
    "CompactionResult",
    "CompactionStrategy",
    "SummarizationStrategy",
    "TokenAwareStrategy",
    # Streaming
    "StreamChunk",
    # Agent Factory
    "AgentBlueprint",
    "AgentConfigurator",
    "AgentFactory",
    "BlueprintValidationError",
    "DynamicCreationDisabledError",
    "OrchestrationBlueprint",
    "OrchestrationFactory",
    "OrchestrationRegistry",
    "OrchestrationSelector",
    "PatternRegistration",
    "SystemPromptGenerator",
    "ToolSelector",
    "create_agent_from_prompt",
    "register_pattern",
    # Workflow Agents
    "ActiveRetrievalAgent",
    "AdaptivePlannerAgent",
    "AgendaAgent",
    "AnalogicalReasoningAgent",
    "AntColonyAgent",
    "AuctionAgent",
    "BacktrackingAgent",
    "BatchAgent",
    "BayesianOptimizationAgent",
    "BeamSearchAgent",
    "BestOfNAgent",
    "BlackboardAgent",
    "BudgetAgent",
    "BufferOfThoughtsAgent",
    "CacheAgent",
    "CascadeAgent",
    "ChainOfAbstractionAgent",
    "ChainOfDensityAgent",
    "CheckpointableAgent",
    "CircuitBreakerAgent",
    "ContractNetAgent",
    "CoVeAgent",
    "CompilerAgent",
    "ConsensusAgent",
    "ContextFilterAgent",
    "CumulativeReasoningAgent",
    "CurriculumAgent",
    "DebateAgent",
    "DelphiMethodAgent",
    "DemonstrateSearchPredictAgent",
    "DifferentialEvolutionAgent",
    "DivideAndConquerAgent",
    "DoubleLoopLearningAgent",
    "DynamicFanOutAgent",
    "EnsembleAgent",
    "EscalationAgent",
    "ExpertPromptingAgent",
    "FeedbackLoopAgent",
    "GeneticAlgorithmAgent",
    "GossipProtocolAgent",
    "GraphOfThoughtsAgent",
    "GroupChatAgent",
    "GuardrailAgent",
    "HandoffAgent",
    "HierarchicalAgent",
    "HypothesisTestingAgent",
    "InnerMonologueAgent",
    "IterativeRetrievalAgent",
    "LeastToMostAgent",
    "LoadBalancerAgent",
    "LoopAgent",
    "MapReduceAgent",
    "MediatorAgent",
    "MemoryConsolidationAgent",
    "MetaOrchestratorAgent",
    "MixtureOfAgentsAgent",
    "MixtureOfExpertsAgent",
    "MixtureOfThoughtsAgent",
    "MonteCarloAgent",
    "MultiArmedBanditAgent",
    "MultiPersonaAgent",
    "NominalGroupAgent",
    "OrchestratorWorkerAgent",
    "ParallelAgent",
    "ParticleSwarmAgent",
    "PipelineParallelAgent",
    "PlannerAgent",
    "PriorityQueueAgent",
    "ProducerReviewerAgent",
    "ProgOfThoughtAgent",
    "PromptChainAgent",
    "RecursiveCriticAgent",
    "RedTeamAgent",
    "ReflexionAgent",
    "RephraseAndRespondAgent",
    "RolePlayingAgent",
    "RouterAgent",
    "SagaAgent",
    "SelfConsistencyAgent",
    "SelfDiscoverAgent",
    "SelfRefineAgent",
    "SequentialAgent",
    "ShadowAgent",
    "SimulatedAnnealingAgent",
    "SkeletonOfThoughtAgent",
    "SkillLibraryAgent",
    "SocraticAgent",
    "SpeculativeAgent",
    "StepBackAgent",
    "SubQuestionAgent",
    "SupervisorAgent",
    "SwarmAgent",
    "TabuSearchAgent",
    "ThreadOfThoughtAgent",
    "TimeoutAgent",
    "TournamentAgent",
    "TreeOfThoughtsAgent",
    "VerifierAgent",
    "VotingAgent",
    "WinnowingAgent",
    # Human-in-the-Loop
    "HumanInputAgent",
    # Runner
    "Runner",
    # Tasker integration
    "json_task_tool",
    "tasker_tool",
    # Skills
    "BaseSkill",
    "SkillDescriptor",
    "SkillRegistry",
    "registry",
    "skill_from_agent",
    "MarkdownSkill",
    "load_skill_md",
    "scan_skills_dir",
    # Tracing
    "LLMCall",
    "TokenUsage",
    "ToolRecord",
    "Trace",
    "TraceCollector",
    "TraceStatus",
    # Memory
    "FileMemoryStore",
    "KeepInMind",
    "MemoryEntry",
    "MemoryStore",
    # Built-in tool collections
    "ALL_TOOLS",
    "DATETIME_TOOLS",
    "TEXT_TOOLS",
    "WEB_TOOLS",
    "PYTHON_TOOLS",
    # ShortFx integration
    "SHORTFX_TOOLS",
    "SHORTFX_DISCOVERY_TOOLS",
    "ShortFxSkill",
    "shortfx_mcp_tools",
    # Workspace
    "CloudStorageEntry",
    "FileEntry",
    "InlineEntry",
    "IODirection",
    "OutputEntry",
    "StorageKind",
    "TemplateEntry",
    "URLEntry",
    "Workspace",
    "WorkspaceEntry",
    # Execution Model (Agent Execution Schema)
    "TaskPacket",
    "EscalationPolicy",
    "ReportingContract",
    "WorkerState",
    "WorkerStateMachine",
    "WorkerTransition",
    "InvalidTransitionError",
    "FailureCategory",
    "FailureClassifier",
    "ClassifiedFailure",
    "RecoveryRecipe",
    "PolicyRule",
    "PolicyResult",
    "PolicyEngine",
    "CallablePolicy",
    "AutoMergePolicy",
    "StaleBranchPolicy",
    "StartupRecoveryPolicy",
    "LaneCompletionPolicy",
    "DegradedModePolicy",
    "VerificationLevel",
    "VerificationContract",
    "VerificationResult",
    "WorktreeManager",
    "WorktreeInfo",
    "WorktreeError",
    "WorkspaceMismatchError",
    "StaleBranchDetector",
    "BranchStatus",
    "ConversationCheckpoint",
    "ConversationCheckpointManager",
    "PlanModeAgent",
    "PlanResult",
]
