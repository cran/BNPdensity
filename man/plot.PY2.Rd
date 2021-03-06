% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/MixPY2.R
\name{plot.PY2}
\alias{plot.PY2}
\title{Plot the density estimate and the 95\% credible interval}
\usage{
\method{plot}{PY2}(x, ...)
}
\arguments{
\item{x}{A fitted object of class PY2}

\item{...}{Further arguments to be passed to generic function, ignored at the moment}
}
\value{
A graph with the density estimate, the 95\% credible interval and a
histogram of the data
}
\description{
Plot the density estimate and the 95\% credible interval
}
\examples{
data(acidity)
out <- MixPY2(acidity, Nit = 50)
plot(out)
}
