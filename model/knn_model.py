from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(
    n_neighbors=7,
    weights="distance",
    metric="minkowski",
    p=2
)
