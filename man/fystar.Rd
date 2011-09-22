\name{fystar}
\alias{fystar}
\title{
Conditional posterior distribution of the distict Ystar
}
\description{
This function evaluates the conditional posterior distribution of the distict 
latents Ystar.
}
\usage{
fystar(v, x, distr.k, sigma.k, distr.p0, mu.p0, sigma.p0)
}
\details{
For internal use.
}
\examples{
## The function is currently defined as
function (v, x, distr, mu, sigma) 
{
    alpha <- p0(v, distr = distr, mu = mu, sigma = sigma)
    Prod <- 1
    for (i in seq(length(x))) {
        fac <- dk(x[i], distr = distr, mu = v, sigma = sigma)
        Prod <- Prod * fac
    }
    f <- alpha * Prod
    return(f)
  }
}
\keyword{internal}
