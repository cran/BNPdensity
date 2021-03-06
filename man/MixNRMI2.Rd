% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/MixNRMI2.R
\name{MixNRMI2}
\alias{MixNRMI2}
\title{Normalized Random Measures Mixture of Type II}
\usage{
MixNRMI2(
  x,
  probs = c(0.025, 0.5, 0.975),
  Alpha = 1,
  Kappa = 0,
  Gama = 0.4,
  distr.k = "normal",
  distr.py0 = "normal",
  distr.pz0 = "gamma",
  mu.pz0 = 3,
  sigma.pz0 = sqrt(10),
  delta_S = 4,
  kappa = 2,
  delta_U = 2,
  Meps = 0.01,
  Nx = 150,
  Nit = 1500,
  Pbi = 0.1,
  epsilon = NULL,
  printtime = TRUE,
  extras = TRUE,
  adaptive = FALSE
)
}
\arguments{
\item{x}{Numeric vector. Data set to which the density is fitted.}

\item{probs}{Numeric vector. Desired quantiles of the density estimates.}

\item{Alpha}{Numeric constant. Total mass of the centering measure.  See
details.}

\item{Kappa}{Numeric positive constant. See details.}

\item{Gama}{Numeric constant. \eqn{0 \leq Gama \leq 1}{0 <= Gama <=1}.  See
details.}

\item{distr.k}{The distribution name for the kernel. Allowed names are "normal", "gamma", "beta", "double exponential", "lognormal" or their common abbreviations "norm", "exp", or an integer number identifying the mixture kernel: 1 = Normal; 2 = Gamma; 3 = Beta; 4 = Double Exponential; 5 = Lognormal.}

\item{distr.py0}{The distribution name for the centering measure for locations. Allowed names are "normal", "gamma", "beta", or their common abbreviations "norm", "exp", or an integer number identifying the centering measure for locations: 1 = Normal; 2 = Gamma; 3 = Beta.}

\item{distr.pz0}{The distribution name for the centering measure for scales.  Allowed names are "gamma", or an integer number identifying the centering measure for
scales: 2 = Gamma. For more options use \code{\link{MixNRMI2cens}}.}

\item{mu.pz0}{Numeric constant. Prior mean of the centering measure for
scales.}

\item{sigma.pz0}{Numeric constant. Prior standard deviation of the centering
measure for scales.}

\item{delta_S}{Numeric positive constant. Metropolis-Hastings proposal
variation coefficient for sampling the scales.}

\item{kappa}{Numeric positive constant. Metropolis-Hastings proposal
variation coefficient for sampling the location parameters.}

\item{delta_U}{Numeric positive constant. Metropolis-Hastings proposal
variation coefficient for sampling the latent U. If `adaptive=TRUE`, `delta_U`is the starting value for the adaptation.}

\item{Meps}{Numeric constant. Relative error of the jump sizes in the
continuous component of the process. Smaller values imply larger number of
jumps.}

\item{Nx}{Integer constant. Number of grid points for the evaluation of the
density estimate.}

\item{Nit}{Integer constant. Number of MCMC iterations.}

\item{Pbi}{Numeric constant. Burn-in period proportion of \code{Nit}.}

\item{epsilon}{Numeric constant. Extension to the evaluation grid range.
See details.}

\item{printtime}{Logical. If TRUE, prints out the execution time.}

\item{extras}{Logical. If TRUE, gives additional objects: means, sigmas,
weights and Js.}

\item{adaptive}{Logical. If TRUE, uses an adaptive MCMC strategy to sample the latent U (adaptive delta_U).}
}
\value{
The function returns a list with the following components:
\item{xx}{Numeric vector. Evaluation grid.}
\item{qx}{Numeric array. Matrix
of dimension \eqn{\texttt{Nx} \times (\texttt{length(probs)} + 1)}{Nx x
(length(probs)+1)} with the posterior mean and the desired quantiles input
in \code{probs}.}
\item{cpo}{Numeric vector of \code{length(x)} with
conditional predictive ordinates.}
\item{R}{Numeric vector of
\code{length(Nit*(1-Pbi))} with the number of mixtures components
(clusters).}
\item{U}{Numeric vector of \code{length(Nit*(1-Pbi))} with the
values of the latent variable U.}
\item{Allocs}{List of
\code{length(Nit*(1-Pbi))} with the clustering allocations.}
\item{means}{List of \code{length(Nit*(1-Pbi))} with the cluster means
(locations). Only if extras = TRUE.}
\item{sigmas}{Numeric vector of
\code{length(Nit*(1-Pbi))} with the cluster standard deviations. Only if
extras = TRUE.}
\item{weights}{List of \code{length(Nit*(1-Pbi))} with the
mixture weights. Only if extras = TRUE.}
\item{Js}{List of
\code{length(Nit*(1-Pbi))} with the unnormalized weights (jump sizes). Only
if extras = TRUE.}
\item{Nm}{Integer constant. Number of jumps of the
continuous component of the unnormalized process.}
\item{delta_Us}{List of
\code{length(Nit*(1-Pbi))} with the sequence of adapted delta_U used in the MH step for the latent variable U.}
\item{Nx}{Integer
constant. Number of grid points for the evaluation of the density estimate.}
\item{Nit}{Integer constant. Number of MCMC iterations.}
\item{Pbi}{Numeric constant. Burn-in period proportion of \code{Nit}.}
\item{procTime}{Numeric vector with execution time provided by \code{proc.time} function.}
\item{distr.k}{Integer corresponding to the kernel chosen for the mixture}
\item{data}{Data used for the fit}
\item{NRMI_params}{A named list with the parameters of the NRMI process}
}
\description{
Bayesian nonparametric estimation based on normalized measures driven
mixtures for locations and scales.
}
\details{
This generic function fits a normalized random measure (NRMI) mixture model
for density estimation (James et al. 2009). Specifically, the model assumes
a normalized generalized gamma (NGG) prior for both, locations (means) and
standard deviations, of the mixture kernel, leading to a fully nonparametric
mixture model.

The details of the model are: \deqn{X_i|Y_i,Z_i \sim
k(\cdot|Y_i,Z_i)}{X_i|Y_i,Z_i ~ k(.|Y_i,Z_i)} \deqn{(Y_i,Z_i)|P \sim P,
i=1,\dots,n}{(Y_i,Z_i)|P ~ P, i=1,...,n} \deqn{P \sim
\textrm{NGG}(\texttt{Alpha, Kappa, Gama; P\_0})}{P ~ NGG(Alpha, Kappa, Gama;
P_0)} where, \eqn{X_i}'s are the observed data, \eqn{(Y_i,Z_i)}'s are
bivariate latent (location and scale) vectors, \code{k} is a parametric
kernel parameterized in terms of mean and standard deviation, \code{(Alpha,
Kappa, Gama; P_0)} are the parameters of the NGG prior with a bivariate
\code{P_0} being the centering measure with independent components, that is,
\eqn{P_0(Y,Z) = P_0(Y)*P_0(Z)}. The parameters of \code{P_0(Y)} are assigned
vague hyper prior distributions and \code{(mu.pz0,sigma.pz0)} are the
hyper-parameters of \code{P_0(Z)}. In particular, \code{NGG(Alpha, 1, 0;
P_0)} defines a Dirichlet process; \code{NGG(1, Kappa, 1/2;P_0)} defines a
Normalized inverse Gaussian process; and \code{NGG(1, 0, Gama; P_0)} defines
a normalized stable process. The evaluation grid ranges from \code{min(x) -
epsilon} to \code{max(x) + epsilon}. By default \code{epsilon=sd(x)/4}.
}
\section{Warning }{
 The function is computing intensive. Be patient.
}

\examples{
\dontrun{
### Example 1
# Data
data(acidity)
x <- acidity
# Fitting the model under default specifications
out <- MixNRMI2(x)
# Plotting density estimate + 95\% credible interval
plot(out)
}

### Example 2
## Do not run
# set.seed(150520)
# data(enzyme)
# x <- enzyme
#  Enzyme2.out <- MixNRMI2(x, Alpha = 1, Kappa = 0.007, Gama = 0.5,
#                          distr.k = "gamma", distr.py0 = "gamma",
#                          distr.pz0 = "gamma", mu.pz0 = 1, sigma.pz0 = 1, Meps=0.005,
#                          Nit = 5000, Pbi = 0.2)
# The output of this run is already loaded in the package
# To show results run the following
# Data
data(enzyme)
x <- enzyme
data(Enzyme2.out)
attach(Enzyme2.out)
# Plotting density estimate + 95\% credible interval
plot(Enzyme2.out)
# Plotting number of clusters
par(mfrow = c(2, 1))
plot(R, type = "l", main = "Trace of R")
hist(R, breaks = min(R - 0.5):max(R + 0.5), probability = TRUE)
# Plotting u
par(mfrow = c(2, 1))
plot(U, type = "l", main = "Trace of U")
hist(U, nclass = 20, probability = TRUE, main = "Histogram of U")
# Plotting cpo
par(mfrow = c(2, 1))
plot(cpo, main = "Scatter plot of CPO's")
boxplot(cpo, horizontal = TRUE, main = "Boxplot of CPO's")
print(paste("Average log(CPO)=", round(mean(log(cpo)), 4)))
print(paste("Median log(CPO)=", round(median(log(cpo)), 4)))
detach()

### Example 3
## Do not run
# set.seed(150520)
# data(galaxy)
# x <- galaxy
#  Galaxy2.out <- MixNRMI2(x, Alpha = 1, Kappa = 0.015, Gama = 0.5,
#                          distr.k = "normal", distr.py0 = "gamma",
#                          distr.pz0 = "gamma", mu.pz0 = 1, sigma.pz0 = 1,  Meps=0.005,
#                          Nit = 5000, Pbi = 0.2)
# The output of this run is already loaded in the package
# To show results run the following
# Data
data(galaxy)
x <- galaxy
data(Galaxy2.out)
attach(Galaxy2.out)
# Plotting density estimate + 95\% credible interval
plot(Galaxy2.out)
# Plotting number of clusters
par(mfrow = c(2, 1))
plot(R, type = "l", main = "Trace of R")
hist(R, breaks = min(R - 0.5):max(R + 0.5), probability = TRUE)
# Plotting u
par(mfrow = c(2, 1))
plot(U, type = "l", main = "Trace of U")
hist(U, nclass = 20, probability = TRUE, main = "Histogram of U")
# Plotting cpo
par(mfrow = c(2, 1))
plot(cpo, main = "Scatter plot of CPO's")
boxplot(cpo, horizontal = TRUE, main = "Boxplot of CPO's")
print(paste("Average log(CPO)=", round(mean(log(cpo)), 4)))
print(paste("Median log(CPO)=", round(median(log(cpo)), 4)))
detach()
}
\references{
1.- Barrios, E., Lijoi, A., Nieto-Barajas, L. E. and Prünster,
I. (2013). Modeling with Normalized Random Measure Mixture Models.
Statistical Science. Vol. 28, No. 3, 313-334.

2.- James, L.F., Lijoi, A. and Prünster, I. (2009). Posterior analysis for
normalized random measure with independent increments. Scand. J. Statist 36,
76-97.

3.- Arbel, J., Kon Kam King, G.,  Lijoi, A., Nieto-Barajas, L.E. and Prüenster, I. (2021). BNPdensity: a package for Bayesian Nonparametric density estimation using Normalised Random Measures with Independent Increments.. Australian and New Zealand Journal of Statistics, to appear
}
\seealso{
\code{\link{MixNRMI2}}, \code{\link{MixNRMI1cens}},
\code{\link{MixNRMI2cens}}, \code{\link{multMixNRMI1}}
}
\author{
Barrios, Kon Kam King, G., E., Lijoi, A., Nieto-Barajas, L.E. and Prüenster, I.
}
\keyword{distribution}
\keyword{models}
\keyword{nonparametrics}
