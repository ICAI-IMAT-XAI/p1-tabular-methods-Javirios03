[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/8usPnGa7)
# Laboratory 01 — Tabular explanation methods

This repository contains the materials for Laboratory Practice 01: Tabular explanation methods. The laboratory is split into a guided notebook that demonstrates methods and an exercise notebook that you must complete. An optional implementation task is provided for extra credit.

## What to do (mandatory)

1. Open and carefully study the notebook `notebook/tabular_explanations.ipynb`.
	- This notebook demonstrates how to explain tabular models using the methods covered in the theory class.
	- It uses common Python libraries (numpy, scikit-learn, etc.) and shows worked examples you will need to replicate and extend.

2. Complete the code and answer the questions in `notebook/exercise.ipynb`.
	- The `exercise.ipynb` notebook contains the tasks you must finish and the questions you must answer.
	- This is the only mandatory part of the laboratory assignment. Make sure your answers are clear and any code cells required by the exercise run correctly.

## Optional (extra credit)

- Implement the methods in `src/shapley_explainer.py` following the Shapley values formulation from the relevant paper and lecture notes.
  - This is optional but will earn extra grade if done correctly and tested.
  - There are unit tests provided in `tests/test_shap.py` that check your implementation against `shap.KernelExplainer` and SHAP additivity. Use these tests to validate correctness.

## Running the notebooks

- Create a new environment using `uv`:

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Open `notebook/tabular_explanations.ipynb` first, in the first cell you can also install the necessary packages for the practice. After studying the notebook, work through `notebook/exercise.ipynb` and save your completed notebook.

## Tests for the optional task

If you attempt the optional Shapley explainer implementation, run the provided tests to verify behavior.

- Note: The tests may train models if saved joblib models are not present in `models/`. Trained models are cached to `models/linear_model.joblib`, `models/random_forest.joblib`, and `models/gradient_boosting.joblib` to speed repeated test runs.

## Grading notes

- Complete `notebook/exercise.ipynb` and provide clear, working answers to all questions: this satisfies the mandatory requirement.
- The Shapley explainer implementation is optional; a correct implementation that passes the tests will receive extra grade.

## Tips and troubleshooting

- If a notebook cell complains about missing packages, make sure your virtual environment is active and the required packages are installed.
- If tests train models and it takes time, let them run once — subsequent runs are faster thanks to model caching in `models/`.

---

Good luck and enjoy the lab!

