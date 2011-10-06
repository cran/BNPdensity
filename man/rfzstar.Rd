\name{rfzstar}
\alias{rfzstar}
\title{
Conditional posterior distribution of the distict Zstar
}
\description{
This function evaluates the ratio of conditional posterior distributions of the distict 
latents Zstar.
}
\usage{
rfzstar(v, x, distr.k, sigma.k, sigma.k2, distr.p0, mu.p0, sigma.p0)
}
\details{
For internal use.
}
\examples{
## The function is currently defined as
function (v, x, distr.k, sigma.k, sigma.k2, distr.p0, mu.p0, sigma.p0) 
{
    alpha <- p0(sigma.k, distr = distr.p0, mu = mu.p0, sigma = sigma.p0)/
             p0(sigma.k2, distr = distr.p0, mu = mu.p0, sigma = sigma.p0)
    Prod <- 1
    for (i in seq(length(x))) {
        fac <- dk(x[i], distr = distr.k, mu = v, sigma = sigma.k)/
               dk(x[i], distr = distr.k, mu = v, sigma = sigma.k2)
        Prod <- Prod * fac
    }
    f <- alpha * Prod
    return(f)
  }
}
\keyword{internal}
