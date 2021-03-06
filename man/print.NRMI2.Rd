% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/MixNRMI2.R
\name{print.NRMI2}
\alias{print.NRMI2}
\title{S3 method for class 'MixNRMI2'}
\usage{
\method{print}{NRMI2}(x, ...)
}
\arguments{
\item{x}{A fitted object of class NRMI2}

\item{...}{Further arguments to be passed to generic function, ignored at the moment}
}
\value{
A visualization of the important information about the object
}
\description{
S3 method for class 'MixNRMI2'
}
\examples{
#' ## Example for censored data
data(acidity)
out <- MixNRMI2(acidity, Nit = 20)
print(out)

data(salinity)
out <- MixNRMI2cens(salinity$left, salinity$right, Nit = 20)
print(out)
}
