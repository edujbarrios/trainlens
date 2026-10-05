# Analyzer plugins

TrainLens keeps the built-in training-session analyzer as the default, but custom analyzers can be registered for process-local extension.

```python
from trainlens import AnalysisConfig, Analyzer, AnalysisResult, register_analyzer, analyze

class MyAnalyzer(Analyzer):
    name = "my_analyzer"

    def analyze(self, snapshot, model, *, config=None):
        return AnalysisResult(summary=["custom analysis"])

register_analyzer(MyAnalyzer())
result = analyze(globals(), config=AnalysisConfig(analyzer="my_analyzer"))
```

Registration is explicit and process-local. Pass `replace=True` only when intentionally replacing an analyzer with the same name. The built-in `training_session` analyzer cannot be unregistered.
