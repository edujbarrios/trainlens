# Python support

TrainLens 0.19.0 supports CPython 3.11, 3.12, 3.13, and 3.14. The package metadata uses `>=3.11,<3.15`, and CI runs the full test, type-check, lint, build, and distribution-check workflow on every supported minor version.

Python 3.15 is not in the advertised support range. Support should be evaluated after its stable release and a successful full CI matrix. All supported Python versions install Matplotlib and ReportLab with the standard TrainLens package.
