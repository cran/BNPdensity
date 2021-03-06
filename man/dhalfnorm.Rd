% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/dhalfnorm.R
\name{dhalfnorm}
\alias{dhalfnorm}
\title{Density half normal}
\usage{
dhalfnorm(x, mean = 0, sd = 1)
}
\description{
Computes the density.
}
\details{
For internal use
}
\examples{

## The function is currently defined as
function(x, mean = 0, sd = 1) {
  ifelse(x < 0, 0, 1) * dnorm(x, mean, sd) / (1 - pnorm(
    0, mean,
    sd
  ))
}
}
\keyword{internal}
