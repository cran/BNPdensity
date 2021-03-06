% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/multMixNRMI.R
\name{multMixNRMI1}
\alias{multMixNRMI1}
\title{Multiple chains of MixNRMI1}
\usage{
multMixNRMI1(
  x,
  probs = c(0.025, 0.5, 0.975),
  Alpha = 1,
  Kappa = 0,
  Gama = 0.4,
  distr.k = "normal",
  distr.p0 = "normal",
  asigma = 0.5,
  bsigma = 0.5,
  delta_S = 3,
  delta_U = 2,
  Meps = 0.01,
  Nx = 150,
  Nit = 1500,
  Pbi = 0.1,
  epsilon = NULL,
  printtime = TRUE,
  extras = TRUE,
  adaptive = FALSE,
  nchains = 4,
  parallel = TRUE,
  ncores = parallel::detectCores()
)
}
\arguments{
\item{x}{Numeric vector. Data set to which the density is fitted.}

\item{probs}{Numeric vector. Desired quantiles of the density estimates.}

\item{Alpha}{Numeric constant. Total mass of the centering measure. See
details.}

\item{Kappa}{Numeric positive constant. See details.}

\item{Gama}{Numeric constant. \eqn{0\leq \texttt{Gama} \leq 1}{0 <= Gama <=
1}.  See details.}

\item{distr.k}{The distribution name for the kernel. Allowed names are "normal", "gamma", "beta", "double exponential", "lognormal" or their common abbreviations "norm", "exp", or an integer number identifying the mixture kernel: 1 = Normal; 2 = Gamma; 3 = Beta; 4 = Double Exponential; 5 = Lognormal.}

\item{distr.p0}{The distribution name for the centering measure. Allowed names are "normal", "gamma", "beta", or their common abbreviations "norm", "exp", or an integer number identifying the centering measure: 1 = Normal; 2 = Gamma; 3 = Beta.}

\item{asigma}{Numeric positive constant. Shape parameter of the gamma prior
on the standard deviation of the mixture kernel \code{distr.k}.}

\item{bsigma}{Numeric positive constant. Rate parameter of the gamma prior
on the standard deviation of the mixture kernel \code{distr.k}.}

\item{delta_S}{Numeric positive constant. Metropolis-Hastings proposal
variation coefficient for sampling sigma.}

\item{delta_U}{Numeric positive constant. Metropolis-Hastings proposal
variation coefficient for sampling the latent U.}

\item{Meps}{Numeric constant. Relative error of the jump sizes in the
continuous component of the process. Smaller values imply larger number of
jumps.}

\item{Nx}{Integer constant. Number of grid points for the evaluation of the
density estimate.}

\item{Nit}{Integer constant. Number of MCMC iterations.}

\item{Pbi}{Numeric constant. Burn-in period proportion of Nit.}

\item{epsilon}{Numeric constant. Extension to the evaluation grid range.
See details.}

\item{printtime}{Logical. If TRUE, prints out the execution time.}

\item{extras}{Logical. If TRUE, gives additional objects: means, weights and
Js.}

\item{adaptive}{Logical. If TRUE, uses an adaptive MCMC strategy to sample the latent U (adaptive delta_U).}

\item{nchains}{The number of chains to run.}

\item{parallel}{Whether to run the chains in parallel. Only works on UNIX-like systems as it rests on Fork parallelism}

\item{ncores}{Number of cores for the parallel run. Defaults to parallel::detectCores(), i.e. the maximum number of cores detected by R on your system.}
}
\value{
a list containing the multiple fits.
}
\description{
Multiple chains of MixNRMI1
}
\examples{

data(acidity)
multMixNRMI1(acidity, parallel = TRUE, Nit = 10, ncores = 2)
}
\seealso{
\code{\link{MixNRMI2}}, \code{\link{MixNRMI1cens}},
\code{\link{MixNRMI2cens}}
}
