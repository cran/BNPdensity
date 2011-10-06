\name{BNPdensity-package}
\alias{BNPdensity-package}
\alias{BNPdensity}
\docType{package}

\title{
    Bayesian nonparametric density estimation
}

\description{ This package performs Bayesian nonparametric density estimation
via a normalized random measure mixture model. The package allows the user to
specify the mixture kernel, the mixing normalized measure and the choice of
performing fully nonparametric mixtures on locations and scales, or
semiparametric mixtures on locations only with common scale parameter.
Options for the kernels are: two kernels with support in the real line
(gaussian and double exponential), two more kernels in the positive line
(gamma and lognormal) and one with bounded support (beta). The options for
the normalized random measures are members of the class of normalized
generalized gamma, which include the Dirichlet process, the normalized
inversed gaussian process and the normalized stable process. }

\details{
    \tabular{ll}{
    Package: \tab BNPdensity\cr
    Type: \tab Package\cr
    Version: \tab 0.7.8\cr
    Date: \tab 2011-10-06\cr
    License: \tab GPL version 2 or later\cr
    LazyLoad: \tab yes\cr
    }
The package includes two main functions: MixNRMI1 and MixNRMI2, which
implement semiparametric mixtures and fully nonparametric mixtures,
respectively. Additionally, the package includes several other functions
required for sampling from conditional distributions in the MCMC
implementation. These functions intended for internal use only.
}

\author{
Barrios, E., Nieto-Barajas, L.E. and Pruenster, I.
Maintainer: <ebarrios@itam.mx>
Ernesto Barrios
}

\references{
    Barrios, E., Nieto-Barajas, L. and Pruenster, I. (2011).
    A study of normalized random measures mixture models. Preprint.
}
\keyword{ package }

\seealso{\code{\link{MixNRMI1}}, \code{\link{MixNRMI2}}}

\examples{
example(MixNRMI1)
example(MixNRMI2)
}
