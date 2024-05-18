This repo is an implementation of stochastic linear unmixing using pytorch.
The data has to be 2d. Rows are independent instances, and can be calculated in a batch without affecting each other's results.
Columns are in the same order between endmembers and observed data to unmix.
Null values are allowed, but a row has to have at least one value that is not null.
