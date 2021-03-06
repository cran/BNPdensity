% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/dhalfcauchy.R
\name{dhalfcauchy}
\alias{dhalfcauchy}
\title{Density half Cauchy}
\usage{
dhalfcauchy(x, location = 0, scale = 1)
}
\description{
Computes the density.
}
\details{
For internal use
}
\examples{

## The function is currently defined as
function(x, location = 0, scale = 1) {
  ifelse(x < 0, 0, 1) * dcauchy(x, location, scale) / (1 - pcauchy(
    0,
    location, scale
  ))
}
}
\keyword{internal}
