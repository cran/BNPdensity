% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/helper_functions.R
\name{compute_thinning_grid}
\alias{compute_thinning_grid}
\title{Compute the grid for thinning the MCMC chain}
\usage{
compute_thinning_grid(Nit, thinning_to = 10)
}
\arguments{
\item{Nit}{Length of the MCMC chain}

\item{thinning_to}{Desired number of iterations after thinning.}
}
\value{
an integer vector of the MCMC iterations retained.
}
\description{
This function creates an real grid then rounds it. If the grid is fine
enough, there is a risk that rounding ties, i.e. iteration which are kept
twice. To avoid this, if the total number of iterations is smaller than
twice the number of iterations desired after thinning, the chain is not
thinned.
}
