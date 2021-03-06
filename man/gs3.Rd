% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/gs3.R
\name{gs3}
\alias{gs3}
\title{Conditional posterior distribution of latent U}
\usage{
gs3(ut, n, r, alpha, kappa, gama, delta)
}
\description{
This function simulates from the conditional posterior distribution of the
latent U.
}
\details{
For internal use.
}
\examples{

## The function is currently defined as
function(ut, n = 200, r = 20, alpha = 1, kappa = 1, gama = 1 / 2,
         delta = 2) {
  w <- ut
  ratio <- NaN
  while (is.nan(ratio)) {
    v <- ustar <- rgamma(1, shape = delta, rate = delta / ut)
    vw <- v / w
    vb <- v + kappa
    wb <- w + kappa
    A <- vw^(n - 2 * delta)
    B <- (vb / wb)^(r * gama - n)
    D <- vb^gama - wb^gama
    E <- 1 / vw - vw
    ratio <- A * B * exp(-alpha / gama * D - delta * E)
  }
  p <- min(1, ratio)
  u <- ifelse(runif(1) <= p, ustar, ut)
  return(u)
}
}
\keyword{internal}
