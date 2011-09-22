\name{MixNRMI2}

\alias{MixNRMI2}

\title{Normalized Random Measures Mixture of Type II}

\description{
    Bayesian nonparametric estimation based on normalized measures driven mixtures
    for locations and scales.
}

\usage{
    MixNRMI2(x, probs = c(0.025, 0.5, 0.975), Alpha = 1, Beta = 0, Gama = 0.4,
    distr.k = 1, distr.py0 = 1, mu.py0 = mean(x), sigma.py0 = 1.5 * sd(x),
    distr.pz0 = 2, mu.pz0 = 1, sigma.pz0 = 10,
    delta = 3, Delta = 2, Nm = 50, Nx = 100, Nit = 1000, Pbi = 0.1,
    epsilon = NULL, printtime = TRUE)
}

\arguments{
  \item{x}{Numeric vector. Data set to which the density is fitted.}
  \item{probs}{Numeric vector. Desired quantiles of the density estimates.}
  \item{Alpha}{Numeric constant. Total mass of the centering measure.
        See details.}
  \item{Beta}{Numeric positive constant. See details.}
  \item{Gama}{Numeric constant. \eqn{0 \leq Gama \leq 1}{0 <= Gama <=1}.
            See details.}
  \item{distr.k}{Integer number identifying the mixture kernel: 1 = Normal;
            2 = Gamma; 3 = Beta; 4 = Double Exponential; 5 = Lognormal.}
  \item{distr.py0}{Integer number identifying the centering measure for
            locations: 1 = Normal; 2 = Gamma; 3 = Beta.}
  \item{mu.py0}{Numeric constant. Prior mean of the centering measure
        for locations.}
  \item{sigma.py0}{Numeric constant. Prior standard deviation of the
            centering measure for locations.}
  \item{distr.pz0}{Integer number identifying the centering measure for scales:
        2 = Gamma, currently the only option.}
  \item{mu.pz0}{Numeric constant. Prior mean of the centering measure for scales.}
  \item{sigma.pz0}{Numeric constant. Prior standard deviation of the
            centering measure for scales.}
  \item{delta}{Numeric positive constant. Metropolis-Hastings proposal
            variation coefficient for sampling the scales.}
  \item{Delta}{Numeric positive constant. Metropolis-Hastings proposal
        variation coefficient for sampling the latent U.}
  \item{Nm}{Integer constant. Number of jumps of the continuous component
        of the unnormalized process.}
  \item{Nx}{Integer constant. Number of grid points for the evaluation of the
        density estimate.}
  \item{Nit}{Integer constant. Number of MCMC iterations.}
  \item{Pbi}{Numeric constant. Burn-in period proportion of \code{Nit}.}
  \item{epsilon}{Numeric constant. Extension to the evaluation grid range.
        See details.}
  \item{printtime}{Logical. If TRUE, prints out the execution time.}
}
\details{
This generic function fits a normalized random measure (NRMI) mixture model for
density estimation (James et al. 2009). Specifically, the model assumes a
normalized generalized gamma (NGG) prior for both, locations (means) and
standard deviations, of the mixture kernel, leading to a fully nonparametric
mixture model.

The details of the model are:
\deqn{X_i|Y_i,Z_i \sim k(\cdot|Y_i,Z_i)}{X_i|Y_i,Z_i ~ k(.|Y_i,Z_i)}
\deqn{(Y_i,Z_i)|P \sim P, i=1,\dots,n}{(Y_i,Z_i)|P ~ P,  i=1,...,n}
\deqn{P \sim \textrm{NGG}(\texttt{Alpha, Beta, Gama; P\_0})}{P ~ NGG(Alpha, Beta, Gama; P_0)}
where, \eqn{X_i}'s are the observed data, \eqn{(Y_i,Z_i)}'s are bivariate
latent (location and scale) vectors, \code{k} is a parametric kernel parameterized
in  terms of mean and standard deviation, \code{(Alpha, Beta, Gama; P_0)} are the
parameters of the NGG prior with a bivariate \code{P_0} being the centering
measure. In particular, \code{NGG(Alpha, 1, 0; P_0)} defines a Dirichlet process;
\code{NGG(1, Beta, 1/2;P_0)} defines a Normalized inverse Gaussian process; and
\code{NGG(1, 0, Gama; P_0)} defines a normalized stable process.
The bivariate measure \code{P_0} is assumed to have independent components,
that is, \eqn{P_0(Y,Z) = P_0(Y)*P_0(Z)}.

The evaluation grid ranges from \code{min(x) - epsilon} to \code{max(x) + epsilon}.
By default \code{epsilon=sd(x)/4}.
}

\value{
    The function returns a list with the following components:
    \item{xx}{Numeric vector. Evaluation grid.}
    \item{qx}{Numeric array. Matrix of dimension
        \eqn{\texttt{Nx} \times (\texttt{length(probs)} + 1)}{Nx x (length(probs)+1)}
        with the posterior mean and the desired quantiles input in \code{probs}.}
    \item{cpo}{Numeric vector of \code{length(x)} with conditional predictive
        ordinates.}
    \item{R}{Numeric vector of \code{length(Nit*(1-Pbi))} with the number of
        mixtures components (clusters).}
    \item{U}{Numeric vector of \code{length(Nit*(1-Pbi))} with the values of the
        latent variable U.}
    \item{Nx}{Integer constant. Number of grid points for the evaluation of the
        density estimate.}
    \item{Nit}{Integer constant. Number of MCMC iterations.}
    \item{Pbi}{Numeric constant. Burn-in period proportion of \code{Nit}.}
    \item{procTime}{Numeric vector with execution time provided by
        \code{proc.time} function.}
}

\references{
1.- Barrios, E., Nieto-Barajas, L.E. and Pruenster, I. (2011). A study of
normalized random measures mixture models. Preprint.


2.- James, L.F., Lijoi, A. and Pruenster, I. (2009). Posterior analysis for
normalized random measure with independent increments. Scand. J. Statist 36,
76-97.
}

\author{Barrios, E., Nieto-Barajas, L.E. and Pruenster, I. }

\section{Warning }{The function is computing intensive. Be patient.}

\seealso{\code{\link{MixNRMI1}}}

\examples{
\dontrun{
### Example 1
# Data
data(acidity)
x <- acidity
# Fitting the model under default specifications
out <- MixNRMI2(x)
# Plotting density estimate + 95% credible interval
attach(out)
m <- ncol(qx)
ymax <- max(qx[,m])
par(mfrow=c(1,1))
hist(x,probability=TRUE,breaks=20,col=grey(.9),ylim=c(0,ymax))
lines(xx,qx[,1],lwd=2)
lines(xx,qx[,2],lty=3,col=4)
lines(xx,qx[,m],lty=3,col=4)
detach()
}

### Example 2
## Do not run
# set.seed(123456)
# data(enzyme)
# x <- enzyme
# Enzyme2.out <- MixNRMI2(x, Alpha = 1, Beta = 0.007, Gama = 0.5, distr.k = 2,
#                 distr.py0 = 2, mu.py0 = 10, sigma.py0 = 10,
#                 distr.pz0 = 2, mu.pz0 = 1, sigma.pz0 = 1,
#                 Nit = 5000, Pbi = 0.2)
# The output of this run is already loaded in the package
# To show results run the following
# Data
data(enzyme)
x <- enzyme
data(Enzyme2.out)
attach(Enzyme2.out)
# Plotting density estimate + 95% credible interval
m <- ncol(qx)
ymax <- max(qx[,m])
par(mfrow=c(1,1))
hist(x,probability=TRUE,breaks=20,col=grey(.9),ylim=c(0,ymax))
lines(xx,qx[,1],lwd=2)
lines(xx,qx[,2],lty=3,col=4)
lines(xx,qx[,m],lty=3,col=4)
# Plotting number of clusters
par(mfrow=c(2,1))
plot(R,type="l",main="Trace of R")
hist(R,breaks=min(R-1):max(R),probability=TRUE)
# Plotting u
par(mfrow=c(2,1))
plot(U,type="l",main="Trace of U")
hist(U,nclass=20,probability=TRUE,main="Histogram of U")
# Plotting cpo
par(mfrow=c(2,1))
plot(cpo,main="Scatter plot of CPO's")
boxplot(cpo,horizontal=TRUE,main="Boxplot of CPO's")
print(paste('Average log(CPO)=',round(mean(log(cpo)),4)))
print(paste('Median log(CPO)=',round(median(log(cpo)),4)))
detach()

### Example 3
## Do not run
# set.seed(123456)
# data(galaxy)
# x <- galaxy
# Galaxy2.out <- MixNRMI2(x, Alpha = 1, Beta = 0.015, Gama = 0.5, distr.k = 1,
#                 distr.py0 = 2, mu.py0 = 20, sigma.py0 = 20,
#                 distr.pz0 = 2, mu.pz0 = 1, sigma.pz0 = 1,
#                 Nit = 5000, Pbi = 0.2)
# The output of this run is already loaded in the package
# To show results run the following
# Data
data(galaxy)
x <- galaxy
data(Galaxy2.out)
attach(Galaxy2.out)
# Plotting density estimate + 95% credible interval
m <- ncol(qx)
ymax <- max(qx[,m])
par(mfrow=c(1,1))
hist(x,probability=TRUE,breaks=20,col=grey(.9),ylim=c(0,ymax))
lines(xx,qx[,1],lwd=2)
lines(xx,qx[,2],lty=3,col=4)
lines(xx,qx[,m],lty=3,col=4)
# Plotting number of clusters
par(mfrow=c(2,1))
plot(R,type="l",main="Trace of R")
hist(R,breaks=min(R-1):max(R),probability=TRUE)
# Plotting u
par(mfrow=c(2,1))
plot(U,type="l",main="Trace of U")
hist(U,nclass=20,probability=TRUE,main="Histogram of U")
# Plotting cpo
par(mfrow=c(2,1))
plot(cpo,main="Scatter plot of CPO's")
boxplot(cpo,horizontal=TRUE,main="Boxplot of CPO's")
print(paste('Average log(CPO)=',round(mean(log(cpo)),4)))
print(paste('Median log(CPO)=',round(median(log(cpo)),4)))
detach()
}

\keyword{distribution}
\keyword{models}
\keyword{nonparametrics}
