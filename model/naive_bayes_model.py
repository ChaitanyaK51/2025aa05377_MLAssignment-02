from sklearn.naive_bayes import GaussianNB

model = GaussianNB(
    var_smoothing=1e-9
)
