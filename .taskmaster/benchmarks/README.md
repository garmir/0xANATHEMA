# LLM Capability Benchmarking Framework for Task Master AI

A comprehensive benchmarking framework for evaluating local LLMs across Task Master AI's core capabilities, providing data-driven insights for optimal model selection and deployment.

## Overview

This framework systematically evaluates local LLMs across 7 core capabilities essential for Task Master AI:

1. **Recursive Task Breakdown** - Decomposing complex tasks into hierarchical subtasks
2. **Multi-Step Reasoning** - Logical reasoning chains and complex problem solving
3. **Context Maintenance** - Maintaining context across long conversations
4. **Code Generation & Analysis** - Writing and understanding code
5. **Research Synthesis** - Combining information from multiple sources
6. **Autonomous Execution Planning** - Creating executable workflows
7. **Meta-Learning** - Learning from feedback and improving performance

## Features

### Comprehensive Evaluation
- **Standardized Benchmarks**: MMLU, HellaSwag, BBH, HumanEval, CodeXGLUE
- **Custom Task Master Tests**: Specialized tests for recursive breakdown, autonomous planning
- **Multi-Dimensional Scoring**: Functional, non-functional, qualitative, and autonomy metrics
- **Performance Monitoring**: Real-time latency, memory usage, and throughput tracking

### Model Support
- **Hugging Face Models**: Transformers with quantization support (4-bit, 8-bit)
- **Ollama Integration**: Local model serving with API compatibility
- **Custom Endpoints**: Support for LocalAI, Text-generation-webui, and custom APIs
- **Quantization Support**: Optimized evaluation for quantized models

### Advanced Analytics
- **Comparative Analysis**: Side-by-side model performance comparison
- **Capability Heatmaps**: Visual representation of model strengths/weaknesses
- **Resource Efficiency**: Performance-to-resource ratio analysis
- **Model Recommendations**: Data-driven selection guidance

## Installation

1. **Clone and Setup**:
```bash
cd /Users/anam/archive/.taskmaster/benchmarks
pip install -r requirements.txt
```

2. **Configure Models** (edit `benchmark_config.json`):
```json
{
  "models": [
    {
      "name": "llama-3.1-70b-awq",
      "model_path": "hugging-quants/Llama-3.1-70B-Instruct-AWQ-INT4",
      "model_type": "huggingface",
      "quantization": "4bit"
    }
  ]
}
```

3. **Install Optional Dependencies**:
```bash
# For Ollama support
pip install ollama-python

# For advanced visualizations
pip install plotly dash

# For model serving
pip install gradio streamlit
```

## Quick Start

### 1. Basic Benchmark Run
```bash
# Run all models with all tests
python llm-capability-benchmark.py --run-all

# Run specific models
python llm-capability-benchmark.py --models llama-3.1-70b-awq mistral-7b-instruct

# Custom output directory
python llm-capability-benchmark.py --run-all --output /path/to/results
```

### 2. Test Suites
```bash
# Run core capabilities only
python run_benchmark_suite.py suite core_capabilities

# Run Task Master specific tests
python run_benchmark_suite.py suite task_master_specific

# Run performance-focused evaluation
python run_benchmark_suite.py suite performance_focused
```

### 3. Capability Comparison
```bash
# Compare models on code generation
python run_benchmark_suite.py capability code

# Compare reasoning capabilities
python run_benchmark_suite.py capability reasoning

# Compare context maintenance
python run_benchmark_suite.py capability context
```

### 4. Quick Evaluation
```bash
# Fast evaluation with reduced test cases
python run_benchmark_suite.py quick

# Quick evaluation for specific models
python run_benchmark_suite.py quick --models llama-3.1-70b-awq
```

## Configuration

### Model Configuration

The `benchmark_config.json` file defines models to benchmark:

```json
{
  "models": [
    {
      "name": "llama-3.1-70b-awq",
      "model_path": "hugging-quants/Llama-3.1-70B-Instruct-AWQ-INT4",
      "model_type": "huggingface",
      "quantization": "4bit",
      "max_tokens": 8192,
      "temperature": 0.7,
      "top_p": 0.9
    },
    {
      "name": "ollama-llama3.1-70b",
      "model_path": "llama3.1:70b",
      "model_type": "ollama",
      "api_endpoint": "http://localhost:11434",
      "max_tokens": 8192
    }
  ]
}
```

### Supported Model Types

1. **Hugging Face Models**:
   - Direct model loading with Transformers
   - Quantization support (4-bit, 8-bit)
   - GPU acceleration when available

2. **Ollama Models**:
   - Local Ollama server integration
   - Custom API endpoints
   - Model pulling and management

3. **Custom APIs**:
   - OpenAI-compatible endpoints
   - LocalAI integration
   - Text-generation-webui support

## Test Framework

### Core Capability Tests

#### 1. Recursive Task Breakdown
Tests the ability to decompose complex tasks into hierarchical subtasks:
- **Input**: High-level task description
- **Expected Output**: Structured JSON with task hierarchy
- **Evaluation**: Accuracy, completeness, logical structure, depth appropriateness

#### 2. Multi-Step Reasoning
Evaluates logical reasoning and problem-solving capabilities:
- **Input**: Complex mathematical or logical problems
- **Expected Output**: Step-by-step solution with final answer
- **Evaluation**: Step accuracy, logical flow, final answer correctness

#### 3. Context Maintenance
Tests ability to maintain context across long conversations:
- **Input**: Multi-turn conversation with context
- **Expected Output**: Contextually appropriate responses
- **Evaluation**: Context retention, answer accuracy, consistency

#### 4. Code Generation & Analysis
Evaluates programming and code understanding capabilities:
- **Input**: Programming task with requirements
- **Expected Output**: Working code with explanation
- **Evaluation**: Correctness, code quality, completeness, error handling

#### 5. Research Synthesis
Tests ability to combine information from multiple sources:
- **Input**: Multiple sources and research question
- **Expected Output**: Comprehensive synthesis
- **Evaluation**: Comprehensiveness, source integration, analysis depth

#### 6. Autonomous Execution Planning
Evaluates autonomous workflow creation capabilities:
- **Input**: Goal and constraints
- **Expected Output**: Detailed execution plan
- **Evaluation**: Plan completeness, feasibility, autonomy level

#### 7. Meta-Learning
Tests ability to learn from examples and apply to new situations:
- **Input**: Learning scenario and new situation
- **Expected Output**: Applied learning and adaptation strategy
- **Evaluation**: Pattern recognition, knowledge transfer, adaptation

### Custom Test Cases

All test cases are defined in `test_data.json` with:
- Detailed scenarios and requirements
- Expected outputs and evaluation criteria
- Complexity ratings and scoring weights
- Comprehensive evaluation rubrics

## Results and Analysis

### Output Structure
```
results/
├── benchmark_results.json          # Raw benchmark data
├── summary_report.json             # High-level summary
├── capability_analysis.csv         # Capability breakdown
├── model_comparison.csv            # Model comparison matrix
├── recommendations.json            # Model selection guidance
├── recommendation_report.md        # Human-readable recommendations
└── benchmark_visualizations.png    # Performance charts
```

### Key Metrics

1. **Functional Metrics**:
   - Accuracy: Correctness of outputs
   - Completeness: Coverage of requirements
   - Consistency: Reliability across tests

2. **Non-Functional Metrics**:
   - Latency: Response time (milliseconds)
   - Memory Usage: Peak memory consumption (MB)
   - Tokens/Second: Throughput measurement

3. **Qualitative Metrics**:
   - Human Evaluation: Expert assessment scores
   - Code Quality: Programming best practices
   - Reasoning Depth: Complexity of reasoning chains

4. **Autonomy Metrics**:
   - Workflow Success Rate: Autonomous task completion
   - Error Recovery: Handling of failure scenarios
   - Adaptation Capability: Learning from feedback

### Visualization

The framework generates comprehensive visualizations:
- Model performance heatmaps
- Capability comparison charts
- Resource efficiency analysis
- Latency and memory usage comparisons

## Model Selection Guidance

### Use Case Recommendations

The framework provides specific recommendations for different use cases:

1. **Overall Best**: Highest average score across all capabilities
2. **Performance Efficiency**: Best score-to-latency ratio
3. **Memory Efficiency**: Best score-to-memory ratio
4. **Balanced Choice**: Optimal balance of performance, speed, and resources

### Capability-Specific Recommendations

- **Code Generation**: Best model for programming tasks
- **Reasoning Tasks**: Optimal for complex problem solving
- **Long Context**: Best for maintaining context across conversations
- **Autonomous Execution**: Ideal for workflow planning and execution

### Implementation Strategy

The framework suggests a phased deployment approach:

1. **Phase 1**: Deploy primary model for general capabilities
2. **Phase 2**: Implement model routing for specialized tasks
3. **Phase 3**: Add performance monitoring and adaptive selection
4. **Phase 4**: Optimize based on production usage patterns

## Integration with Task Master AI

### Compatibility

The benchmarking framework is designed to integrate seamlessly with:
- Task Master AI's existing evaluation pipeline
- Local LLM serving infrastructure (Ollama, LocalAI)
- Task decomposition and planning modules
- Autonomous workflow execution systems

### Migration Support

Results provide data-driven guidance for:
- Selecting optimal local LLMs for Task Master migration
- Configuring model serving infrastructure
- Setting up model routing and fallback strategies
- Establishing performance monitoring and alerting

## Advanced Usage

### Custom Test Development

Add new tests by extending the `BenchmarkTest` base class:

```python
class CustomCapabilityTest(BenchmarkTest):
    def __init__(self):
        super().__init__("custom_test", "Custom Capability")
    
    async def run(self, model: LLMInterface) -> BenchmarkResult:
        # Implement custom test logic
        pass
```

### Model Interface Extension

Support new model types by implementing the `LLMInterface`:

```python
class CustomModelInterface(LLMInterface):
    async def generate(self, prompt: str, max_tokens: int = 512) -> str:
        # Implement model-specific generation logic
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        # Return model metadata
        pass
    
    def cleanup(self):
        # Cleanup resources
        pass
```

### Continuous Benchmarking

Set up automated benchmarking for continuous model evaluation:

```bash
# Daily benchmarking
crontab -e
0 2 * * * cd /path/to/benchmarks && python run_benchmark_suite.py quick

# Weekly comprehensive evaluation
0 1 * * 0 cd /path/to/benchmarks && python run_benchmark_suite.py selection
```

## Performance Optimization

### Resource Management

- **Memory Optimization**: Use quantized models and gradient checkpointing
- **GPU Utilization**: Automatic device placement and mixed precision
- **Parallel Execution**: Concurrent test execution for faster results
- **Caching**: Result caching to avoid redundant evaluations

### Scalability

- **Distributed Evaluation**: Support for multi-GPU and multi-node execution
- **Batch Processing**: Efficient batch evaluation for large model sets
- **Incremental Updates**: Delta evaluation for model updates
- **Result Persistence**: Efficient storage and retrieval of benchmark data

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce model size or use quantization
   - Decrease batch size or max_tokens
   - Use CPU fallback for evaluation

2. **Model Loading Failures**:
   - Verify model paths and availability
   - Check network connectivity for downloads
   - Ensure sufficient disk space

3. **API Connection Issues**:
   - Verify Ollama or API server is running
   - Check endpoint URLs and authentication
   - Review firewall and network settings

### Debug Mode

Enable detailed logging for troubleshooting:

```python
logging.basicConfig(level=logging.DEBUG)
python llm-capability-benchmark.py --debug
```

## Contributing

### Adding New Tests

1. Define test cases in `test_data.json`
2. Implement test class extending `BenchmarkTest`
3. Add test to appropriate suite in `run_benchmark_suite.py`
4. Update documentation and examples

### Model Support

1. Implement `LLMInterface` for new model type
2. Add configuration schema to `benchmark_config.json`
3. Update model factory in `BenchmarkRunner`
4. Add integration tests and documentation

## License and Citation

This benchmarking framework is part of the Task Master AI project. When using this framework in research or production, please cite:

```
Task Master AI LLM Capability Benchmarking Framework
Version: 1.0.0
URL: https://github.com/your-org/task-master-ai
Date: 2025-07-10
```

## Support

For issues, questions, or contributions:
- Create issues in the project repository
- Review existing documentation and examples
- Join the Task Master AI community discussions
- Contact the development team for enterprise support

---

*Last updated: July 10, 2025*