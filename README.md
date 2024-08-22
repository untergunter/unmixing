This repo is an implementation of stochastic linear unmixing using pytorch.\n
It is a fast and efficient, and can unmix even if there are null values.\n
\n
$H$ is a 2d metrix, each row in it is a single end member.\n
Each column is a single feature of the end members.\n
No null values are allowed in $H$.\n
$O$ is a 2d metrix, each row a single observed instance.\n
The columns of $O$ are the same as the columns of $H$.\n
Null values are allowed in $O$.\n
The main class of this repo, LinearUnmixing, is calculating $R$ matrix\n
that minimizes the loss between $O$ and $H \cdot R$.\n
$R$ is constrained to be non-negative and each row in R sums to 1.\n
Reconstruction error can be used for filtering, since there is no\n
assessment for $R$, but we can asses the distance between $O$ and $H \cdot R$.\n
\n
requirements:\n
numpy\n
torch\n
\n
Installation:\n
pip install unmixing\n
\n
Usage example:\n

    n_endmembers = 5
    n_features = 4
    n_rows = 1_000

    endmembers = torch.randint(1, 10, (n_endmembers, n_features), dtype=torch.float32)
    non_negative = torch.randn((n_rows, n_endmembers)) ** 2
    rates = non_negative / non_negative.sum(axis=1).reshape((-1, 1))
    observed = torch.matmul(rates, endmembers)
    lm = LinearUnmixing(endmembers, observed)
    lm.unmix()
    pred_rates = lm.get_rates()

    rates_errors = (pred_rates - rates.numpy()).sum(axis=1).round(2)
    print(f"first example - max rate error: {rates_errors.max()}")

    reconstruction_error_per_observation = lm.get_errors()
    print(f"first example - mean absolute reconstruction error: "
          f"{np.mean(np.abs(reconstruction_error_per_observation))}")

    detailed_reconstruction_error = lm.get_errors_per_column()
    print(f"first example - mean absolute reconstruction error per entry: "
          f"{np.mean(np.abs(detailed_reconstruction_error))}")