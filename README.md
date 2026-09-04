# COMP5318 Assignment 1 — Rice Classification

Group assignment: build a full classification pipeline (pre-processing, 8 classifiers,
hyperparameter tuning, evaluation) on the Rice dataset.

**Due: 11:59 pm Friday 18 September 2026.** Late penalty 5%/day, nothing accepted after 3 days.

## Running the project

Requires [uv](https://docs.astral.sh/uv/). Python 3.13 is pinned in `.python-version`.

```bash
# 1. Install the exact dependencies from uv.lock (first time, and after anyone runs `uv add`)
uv sync

# 2. Launch JupyterLab and open ml-project-group.ipynb
uv run jupyter lab
```

Useful one-offs:

```bash
# Execute the whole notebook headlessly to check it runs top-to-bottom (writes a copy, leaves the original alone)
uv run jupyter nbconvert --to notebook --execute --output /tmp/check.ipynb ml-project-group.ipynb

# Run a throwaway script inside the project environment
uv run python somescript.py

# Add a library (updates pyproject.toml + uv.lock — commit both, then everyone runs `uv sync`)
uv add <package>
```

Always go through `uv run`. The system Python on this machine has no pandas, numpy or
scikit-learn, so a bare `python3 script.py` will fail on the imports.

## Layout

```
code/
├── ml-project-group.ipynb   # the deliverable (rename to ml-project-group<N>.ipynb once we have our group number)
├── rice-final2.csv          # the dataset we are marked on
├── test-before.csv          # 209 x 6 — runnability check only, NOT for marking
├── pyproject.toml           # dependencies
└── uv.lock                  # pinned versions, keeps the three of us identical
```

The pristine Canvas downloads and the assignment PDF live one directory up in `../`.

## Work split

| Person | Owns |
|---|---|
| **A — Integrator** | §1 pre-processing, shared `cvKFold` / `train_test_split` / `run_grid_search()`, the results block, `test-before.csv` check, master notebook, PDF export, submission |
| **B — Baselines** | Logistic Regression, Naive Bayes (Part 1); KNN, Decision Tree, SVM (Part 2). Consolidates the reflection cell |
| **C — Ensembles** | AdaBoost, Gradient Boosting, Random Forest (+ macro & weighted F1) |

**Only Person A edits `ml-project-group.ipynb`.** Notebook merges are destructive — B and C
develop in their own scratch copies and hand over cell contents.

## TODO

### Person A — shared scaffolding
- [x] Cell 3 — imports
- [x] Cell 5 — `load_dataset()`, reads `?` as NaN
- [x] Cell 6 — `preprocess_dataset()`: mean imputation, min-max scaling, class1/class2 -> 0/1
- [x] Cell 7 — `print_data(X, y)`, first 10 rows to 4dp
- [ ] Cell 1 — fill in group number and the three SIDs (**no names** — marking is anonymous)
- [ ] Cell 10 — `train_test_split(X, y, stratify=y, random_state=0)` (**decide `test_size` first**)
- [ ] Cell 10 — `run_grid_search(estimator, param_grid)` helper returning `(best_params, best_cv_score, test_accuracy)`
- [ ] Cell 23 — assemble everyone's results, template's exact print labels, all floats `.4f`
- [ ] Cell 25 — run everything against `test-before.csv`, then **clear the output before submitting**
- [ ] Rename notebook to include the group number
- [ ] Restart & Run All on rice, export `.ipynb` + `.pdf`, submit to the two separate Canvas boxes

### Person B — Part 1 + simple models
- [ ] Cell 11 — Logistic Regression, `random_state=0`, `cross_val_score(..., cv=cvKFold)`, report the mean
- [ ] Cell 12 — Naive Bayes, `GaussianNB()` (no `random_state` — it has no such parameter)
- [ ] Cell 16 — KNN grid: `k=[1,3,5,7]`, `p=[1,2]`
- [ ] Cell 17 — Decision Tree, `criterion='entropy'`, `random_state=0`; `max_depth`, `min_samples_split`, `min_samples_leaf`
- [ ] Cell 21 — SVM, `random_state=0`; `C`, `gamma`
- [ ] Cell 27 — collect everyone's paragraphs into the reflection

### Person C — ensembles
- [ ] Cell 18 — AdaBoost, `random_state=0`; `n_estimators`, `learning_rate`
- [ ] Cell 19 — Gradient Boosting, `random_state=0`; `max_depth`, `n_estimators`, `learning_rate` (48 combos x 10 folds — start early, use `n_jobs=-1`)
- [ ] Cell 20 — Random Forest, `criterion='entropy'`, `max_features='sqrt'`, `random_state=0`; `n_estimators`, `max_leaf_nodes`
- [ ] Cell 20 — also report test-set **macro average F1** and **weighted average F1**

### Everyone
- [ ] Join the same Canvas group (People -> Machine Learning Project Group)
- [ ] Write 2-3 sentences on your own classifiers for the reflection: how they compare, and what
      hyperparameter tuning did to their performance (both points are explicitly required)
- [ ] Fill in the AI Acknowledgement cell

## Open decisions

- **`test_size` for `train_test_split`** — the brief never states one. Default to sklearn's 0.25
  unless Canvas or Ed says otherwise. All three of us must use the same value or the Part 2
  numbers are not comparable.

## Conventions — these are marked

1. **Never hardcode 1400 or 7.** Markers re-run this on an unseen `unknown.csv` with a different
   number of features and rows. Always slice with `iloc[:, :-1]` / `iloc[:, -1]`.
   `test-before.csv` has 6 features and exists to prove this works.
2. **`random_state=0` everywhere** it exists — classifiers, `train_test_split`, `StratifiedKFold`.
3. **All reported floats to 4 decimal places** (`"{:.4f}"`). Integers such as `k`, `p`,
   `n_estimators` and `max_leaf_nodes` stay integers.
4. **Keep the template's exact print labels** and keep `print_data` byte-for-byte as provided.
5. **If a cell errors, that part scores 0.** Restart & Run All before every handover.
6. The submitted notebook must show **rice results only** — no `test-before.csv` output.

## Note for the reflection

Section 1 imputes and scales the *whole* dataset before the train/test split, because the spec
requires printing the pre-processed data up front. Strictly this leaks test-set statistics into
the scaler; a `Pipeline` fitted per-fold would be cleaner. Worth calling out in the discussion —
it reads as understanding rather than as a mistake.
