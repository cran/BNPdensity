% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/pt_.R
\name{pt_}
\alias{pt_}
\title{Distribution function non-standard student-t}
\usage{
pt_(x, df, mean, sd)
}
\description{
Computes the cdf.
}
\details{
For internal use
}
\examples{

## The function is currently defined as
function(x, df, mean, sd) {
  pt((x - mean) / sd, df, ncp = 0)
}
}
\keyword{internal}
