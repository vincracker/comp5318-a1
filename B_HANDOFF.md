# Person B handoff

Person B completed the following assigned work in `main.ipynb` on branch `B`:

- Cell 16: KNN with `GridSearchCV`
- Cell 17: Decision Tree with `criterion='entropy'` and `GridSearchCV`
- Cell 19: Gradient Boosting with `GridSearchCV`
- Cell 27: reflection draft for the three models

## Shared setup included for independent execution

Cell 16 also contains `train_test_split` and `run_grid_search()` so this branch can be
executed independently. The split uses `stratify=y`, `random_state=0`, and sklearn's
default test size (0.25), as agreed in the README. Person A can reuse this implementation
or keep the equivalent shared helper in the master notebook.

## Rice results

| Classifier | Best parameters | CV accuracy | Test accuracy |
|---|---|---:|---:|
| KNN | `n_neighbors=5`, `p=1` | 0.9371 | 0.9257 |
| Decision Tree | `max_depth=5`, `min_samples_split=10`, `min_samples_leaf=4` | 0.9371 | 0.9171 |
| Gradient Boosting | `max_depth=1`, `n_estimators=50`, `learning_rate=0.3` | 0.9448 | 0.9457 |

## Validation

- `main.ipynb` on branch `B` was restarted and executed from top to bottom with no errors.
- The same code was executed with `test-before.csv` (209 examples and 6 features) with
  no errors, confirming that the implementation does not assume seven features.
- The committed notebook contains only the required `rice-final2.csv` outputs.

## Integration notes for Person A

Review the changes to cells 16, 17, 19, and the relevant paragraphs in cell 27 when merging
branch `B` into the master notebook. Keep only one shared train/test split and one shared
grid-search helper. After all eight classifiers are merged, revise the final reflection so
that its overall comparison includes the other five models as well.
