# COMP5318 Assignment 1 — Rice Classification

Group assignment: build a full classification pipeline (pre-processing, 8 classifiers,
hyperparameter tuning, evaluation) on the Rice dataset.

**Due: 11:59 pm Friday 18 September 2026.** Late penalty 5%/day, nothing accepted after 3 days.

## Running the project

Requires [uv](https://docs.astral.sh/uv/). Python 3.13 is pinned in `.python-version`.

```bash
# 1. Install the exact dependencies from uv.lock (first time, and after anyone runs `uv add`)
uv sync

# 2. Launch JupyterLab and open main.ipynb
uv run jupyter lab
```

Useful one-offs:

```bash
# Execute the whole notebook headlessly to check it runs top-to-bottom (writes a copy, leaves the original alone)
uv run jupyter nbconvert --to notebook --execute --output /tmp/check.ipynb main.ipynb

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
├── main.ipynb               # the deliverable (rename to ml-project-group<N>.ipynb before submitting)
├── rice-final2.csv          # the dataset we are marked on
├── test-before.csv          # 209 x 6 — runnability check only, NOT for marking
├── pyproject.toml           # dependencies
└── uv.lock                  # pinned versions, keeps the three of us identical
```

The pristine Canvas downloads and the assignment PDF live one directory up in `../`.

## Work split

Three classifiers each. Cell numbers refer to the template's numbering.

| Person | Classifiers | Also owns |
|---|---|---|
| **A** | Logistic Regression (11), Naive Bayes (12), AdaBoost (18) | §1 pre-processing, shared `cvKFold` / `train_test_split` / `run_grid_search()` (10), results block (23), `test-before.csv` check (25), master notebook, PDF export, submission |
| **B** | KNN (16), Decision Tree (17), Gradient Boosting (19) | Consolidates the reflection (27) |
| **C** | SVM (21), Random Forest + macro & weighted F1 (20) | — |

**Only Person A edits `main.ipynb`.** Notebook merges are destructive — B and C
develop in their own scratch copies and hand over cell contents.

Settings that are marked, per classifier:

- **Part 1** (Logistic Regression, Naive Bayes) — no tuning. `cross_val_score(..., cv=cvKFold)`,
  report the mean. `GaussianNB()` takes no `random_state`; passing one raises.
- **Part 2** (the other six) — `GridSearchCV` over the training split, then score the best
  estimator on the held-out test set.
- Decision Tree: `criterion='entropy'`. Random Forest: `criterion='entropy'`, `max_features='sqrt'`.
- Random Forest also reports test-set **macro average F1** and **weighted average F1**.
- Gradient Boosting is the heaviest grid (48 combos x 10 folds) — start early, use `n_jobs=-1`.

### Everyone

- Join the same Canvas group (People -> Machine Learning Project Group).
- Write 2-3 sentences on your own classifiers for the reflection: how they compare, and what
  hyperparameter tuning did to their performance (both points are explicitly required).
- Fill in the AI Acknowledgement cell.

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
7. **Rename `main.ipynb` to `ml-project-group<N>.ipynb` before submitting** — the spec requires
   the group number in both the `.ipynb` and the `.pdf` filename.

## Note for the reflection

Section 1 imputes and scales the *whole* dataset before the train/test split, because the spec
requires printing the pre-processed data up front. Strictly this leaks test-set statistics into
the scaler; a `Pipeline` fitted per-fold would be cleaner. Worth calling out in the discussion —
it reads as understanding rather than as a mistake.
