# Analyzer plugin API

TrainLens uses a process-local analyzer registry so custom notebook analysis can be added without modifying the built-in training analyzer.

```python
from trainlens import AnalysisConfig, Analyzer, AnalysisResult, analyze, register_analyzer

class MyAnalyzer(Analyzer):
    name = "my_analyzer"

    def analyze(self, snapshot, model, *, config=None):
        return AnalysisResult(summary=["custom analysis"])

register_analyzer(MyAnalyzer())
result = analyze(globals(), config=AnalysisConfig(analyzer="my_analyzer"))
```

Registration is explicit. Use `replace=True` only when intentionally replacing an analyzer with the same name. The built-in `training_session` analyzer is protected from unregistration.
