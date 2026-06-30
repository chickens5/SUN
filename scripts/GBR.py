# ============================================================
# GRADIENT BOOSTING REGRESSOR — Reference Implementation
# Hyperparameters match PipelineConfig defaults in ggsp_pipeline_v7.py
# ============================================================

from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(
    
    # n_estimators = 200
    # Number of boosting stages (sequential trees).
    # Iteration i fits a shallow tree to the residuals of all previous trees.
    # 200 is a practical balance between fit quality and training time on ~30k samples.
    n_estimators=200,

    # max_depth = 3
    # Each tree has at most 3 levels → 8 leaf regions.
    # Allows three-way feature interactions (e.g., speed AND Bz AND prior Kp)
    # without the overfitting risk of depth 4+.
    max_depth=3,

    # learning_rate = 0.05
    # Shrinkage factor: each tree contributes only 5% of its fitted correction.
    # Smoother convergence, more robust across different solar-cycle amplitudes.
    learning_rate=0.05,

    # subsample = 0.8
    # Stochastic boosting: each tree is trained on a random 80% of samples.
    # Adds regularisation and reduces variance without hurting accuracy much.
    subsample=0.8,

    # min_samples_split = 20
    # A node must have at least 20 samples before it can split.
    # Prevents the tree from making splits on rare extreme-Kp events in the training set.
    min_samples_split=20,

    # min_samples_leaf = 5
    # Each leaf must contain at least 5 training points.
    # Guards against overfitting to individual storm observations.
    min_samples_leaf=5,

    # random_state = 42
    # Fixed seed for subsample randomisation → fully reproducible runs.
    random_state=42,
)

# ── Training ──────────────────────────────────────────────────────────────────
# model.fit(X_train, y_train)
#
# Internally:
#   1. Initialise prediction = mean(y_train)
#   2. For i in range(200):
#        residual_i = y_train - current_prediction
#        fit tree_i to (X_train, residual_i) with max_depth=3
#        current_prediction += 0.05 * tree_i.predict(X_train)
#   3. Store all 200 trees for inference

# ── Prediction ────────────────────────────────────────────────────────────────
# y_pred = model.predict(X_test)
#
# Conceptually:
#   y_pred = mean(y_train)
#            + 0.05 * tree_0.predict(X_test)
#            + 0.05 * tree_1.predict(X_test)
#            + ...
#            + 0.05 * tree_199.predict(X_test)

    
    # n_estimators = 200
    # ==================
    # Number of boosting stages (sequential trees).
    #
    # How it works:
    # Iteration 0: Fit tree_0 to y_train
    # Iteration 1: Fit tree_1 to RESIDUALS of tree_0 (what tree_0 got wrong)
    # Iteration 2: Fit tree_2 to RESIDUALS of (tree_0 + tree_1)
    # ...
    # Iteration 199: Fit tree_199 to residuals of (sum of all previous)
    #
    # More iterations → better fit but risk of overfitting
    # 200 chosen as compromise for 1,300 training samples
    
    n_estimators=200,
    
    
    # max_depth = 3
    # =============
    # Each individual tree has max depth 3 (4 leaf levels).
    # Shallow trees learn simple patterns, prevent overfitting.
    #
    # Tree with depth 3:
    #              [Feature X > 500?]
    #              /                 \
    #         [Depth 1]          [Depth 1]
    #         /        \         /        \
    #      [D2]      [D2]     [D2]      [D2]
    #     / | \ \   / | \ \  / | \ \  / | \ \
    #    [L][L][L][L][L][L][L][L][L][L][L][L]  [Depth 3 — Leaves]
    #
    # Depth 3 can create 2^3 = 8 leaf regions, enough to model nonlinearity
    # but not so deep as to overfit individual training points
    
    max_depth=3,
    
    
    # learning_rate = 0.05
    # ====================
    # Also called "shrinkage" or "eta".
    #
    # Update rule at iteration i:
    # prediction_i = prediction_{i-1} + learning_rate * tree_i_prediction
    #
    # Small learning_rate (0.05 = 5%):
    # - Each tree contributes only 5% of its fitted value
    # - Requires MORE iterations to fit
    # - Smoother learning curve, less overfitting
    # - More robust to noise
    #
    # Mathematical effect:
    # final_pred = sum_{i=0}^{n_estimators} (0.05 * tree_i_pred)
    #
    # Without shrinkage (lr=1.0):
    # final_pred = sum_{i=0}^{n_estimators} (1.0 * tree_i_pred)
    # This overfits faster
    
    learning_rate=0.05,
    
    
    # subsample = 0.8
    # ===============
    # Stochastic boosting: each tree sees only 80% of training data
    # (randomly sampled without replacement at each iteration).
    #
    # Why?
    # - Reduces variance (ensemble of models trained on different subsets)
    # - Speeds up training (only 80% of computation)
    # - Adds regularization (prevents memorizing all outliers)
    #
    # Example:
    # Train set has 1,000 samples. Tree_0 gets random 800 samples.
    # Tree_1 gets different random 800 samples. Etc.
    # This diversity reduces overfitting.
    
    subsample=0.8,
    
    
    # min_samples_split = 10
    # ======================
    # Minimum number of samples required at a node to consider splitting.
    #
    # If node has < 10 samples, it becomes a leaf (no further split).
    #
    # Prevents overfitting to rare training instances.
    # Example:
    # - Node A has 50 samples: Can split (50 >= 10)
    # - Node B has 3 samples: Cannot split; stays as leaf
    
    min_samples_split=10,
    
    
    # random_state = 42
    # =================
    # Seed for NumPy random number generator.
    # Ensures reproducibility: same seed → same random choices → same model
    # Without this, model would vary each run due to subsample randomness
    
    random_state=42
)

# ============================================================
# TRAINING
# ============================================================

# Fit the model: learn patterns in 1,000+ training samples
model.fit(X_train, y_train)

# Behind the scenes, model.fit does:
# 1. Initialize prediction with mean(y_train)
# 2. FOR iteration 0 TO n_estimators-1:
#    a. Compute residuals: residual_i = y_train - current_prediction
#    b. Fit shallow tree (max_depth=3) to (X_train, residual_i)
#    c. Get tree's prediction on training data: tree_pred_i
#    d. Update: current_prediction += 0.05 * tree_pred_i
# 3. Store all 200 trees for later prediction

# ============================================================
# PREDICTION
# ============================================================

y_pred = model.predict(X_test)

# Behind the scenes, model.predict does:
# 1. Initialize prediction with mean(y_train)
# 2. FOR each of 200 trees:
#    a. Get tree's prediction on X_test
#    b. Add: prediction += 0.05 * tree_prediction
# 3. Return final prediction

# Mathematical form (conceptually):
# y_pred = mean(y_train) + 0.05 * tree_0.predict(X_test) 
#                        + 0.05 * tree_1.predict(X_test)
#                        + ...
#                        + 0.05 * tree_199.predict(X_test)
#
# This is an ensemble: final prediction is WEIGHTED AVERAGE of all 200 trees