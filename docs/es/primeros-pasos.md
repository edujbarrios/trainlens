# Instalación y guía rápida

## Instalación

```bash
python -m pip install trainlens
```

Para exportar PDF:

```bash
python -m pip install "trainlens[pdf]"
```

Para desarrollar localmente:

```bash
git clone https://github.com/edujbarrios/trainlens.git
cd trainlens
python -m pip install -e ".[dev]"
python -m pytest
```

## Flujo de cinco minutos en Jupyter

Entrena un modelo o conserva un historial métrico en el cuaderno:

```python
history = {
    "train_loss": [0.61, 0.50, 0.25],
    "eval_loss": [0.62, 0.52, 0.28],
    "accuracy": [0.89, 0.91, 0.92],
    "val_accuracy": [0.82, 0.83, 0.92],
}
```

Configura un proveedor LLM y carga la extensión:

```python
%env TRAINLENS_LLM_BASE_URL=https://api.openai.com/v1
%env TRAINLENS_LLM_API_KEY=reemplazar
%env TRAINLENS_LLM_MODEL=tu-modelo
%load_ext trainlens.magic.extension
%explain_training
%suggest_improvements
```

`%explain_training` también captura las métricas en memoria. Ejecútalo tras un
segundo experimento y usa `%compare_runs`. Este historial pertenece a la
extensión cargada y no es una base de datos persistente.

## Primer flujo sin LLM

Las comparaciones, la monitorización, las recomendaciones y la exportación no
necesitan un LLM:

```python
from trainlens import compare_runs, render_run_comparison

comparison = compare_runs(
    {"validation_loss": 0.52, "accuracy": 0.84},
    {"validation_loss": 0.47, "accuracy": 0.87},
    baseline_name="baseline",
    experiment_name="learning rate menor",
)
print(render_run_comparison(comparison))
```

Continúa con [cuadernos.md](cuadernos.md) para la inspección automática o con
[api-python.md](api-python.md) para llamadas explícitas y comprobables.

