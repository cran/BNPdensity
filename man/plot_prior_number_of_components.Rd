% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/expected_number_of_clusters_stable_process.R
\name{plot_prior_number_of_components}
\alias{plot_prior_number_of_components}
\title{This plots the prior distribution on the number of components for the stable
process. The Dirichlet process is provided for comparison.}
\usage{
plot_prior_number_of_components(
  n,
  Gama,
  Alpha = 1,
  grid = NULL,
  silence = TRUE
)
}
\arguments{
\item{n}{Number of data points}

\item{Gama}{Numeric constant. 0 <= Gama <=1.}

\item{Alpha}{Numeric constant. Total mass of the centering measure for the
Dirichlet process.}

\item{grid}{Integer vector. Level of truncation when computing the expectation. Defaults to
n. If greater than n, it is fixed to n.}

\item{silence}{Boolean. Whether to print the current calculation step for the Stable process, as the function can be long}
}
\value{
A plot with the prior distribution on the number of components.
}
\description{
This plots the prior distribution on the number of components for the stable
process. The Dirichlet process is provided for comparison.
}
\examples{

plot_prior_number_of_components(50, 0.4)
}
