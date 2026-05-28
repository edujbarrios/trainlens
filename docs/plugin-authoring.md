# Plugin Authoring

TrainLens analyzers implement the `Analyzer` interface and can be registered with `AnalyzerRegistry`.

```python
from trainlens.analyzers.base import Analyzer


class MyAnalyzer(Analyzer):
    name = "my_analyzer"

    def analyze(self, snapshot, model):
        ...
```

Plugins should return evidence-backed `AnalysisResult` objects and avoid hidden network calls.
