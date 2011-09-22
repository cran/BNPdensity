\name{gs3}
\alias{gs3}
\title{
Conditional posterior distribution of latent U
}
\description{
This function simulates from the conditional posterior distribution of the latent U.
}
\usage{
gs3(ut, n, r, alpha, beta, gama, delta)
}
\details{
For internal use. 
}
\examples{
## The function is currently defined as
function (ut, n = 200, r = 20, alpha = 1, beta = 1, gama = 1/2, 
    delta = 2) 
{
    w <- ut
    ratio <- NaN
    while (is.nan(ratio)) {
        v <- ustar <- rgamma(1, shape = delta, rate = delta/ut)
        vw <- v/w
        vb <- v + beta
        wb <- w + beta
        A <- vw^(n - 2 * delta)
        B <- (vb/wb)^(r * gama - n)
        D <- vb^gama - wb^gama
        E <- 1/vw - vw
        ratio <- A * B * exp(-alpha/gama * D - delta * E)
    }
    p <- min(1, ratio)
    u <- ifelse(runif(1) <= p, ustar, ut)
    return(u)
  }
}
\keyword{internal}