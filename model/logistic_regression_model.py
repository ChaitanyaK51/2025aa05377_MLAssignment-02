from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    max_iter=1000,
    C=0.5,
    solver="lbfgs",
    penalty="l2",
    random_state=42
)
