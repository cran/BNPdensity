\name{gs4}
\alias{gs4}
\title{
Resampling Ystar function
}
\description{
This funciton resamples the distinct Ystar in the semiparametric model. 
}
\usage{
gs4(ystar, x, idx, distr.k, sigma.k, distr.p0, mu.p0, sigma.p0)
}
\details{
For internal use. 
}
\examples{
## The function is currently defined as
function (ystar, x, idx, distr.k = 1, sigma.k = 1, distr.p0 = 1, 
    mu.p0 = 0, sigma.p0 = 1) 
{
    r <- length(ystar)
    nstar <- as.numeric(table(idx))
    for (j in seq(r)) {
        id <- which(!is.na(match(idx, j)))
        xj <- x[id]
        xbar <- sum(xj)/nstar[j]
        y2star <- rk(1, distr = distr.k, mu = xbar, sigma = sigma.k/sqrt(nstar[j]))
        f.ratio <- fystar(y2star, xj, distr = distr.k, sigma = sigma.k, 
            distr.p0 = distr.p0, mu.p0 = mu.p0, sigma.p0 = sigma.p0)/fystar(ystar[j], 
            xj, distr = distr.k, sigma = sigma.k, distr.p0 = distr.p0, 
            mu.p0 = mu.p0, sigma.p0 = sigma.p0)
        k.ratio <- dk(ystar[j], distr = distr.k, mu = xbar, sigma = sigma.k/sqrt(nstar[j]))/dk(y2star, 
            distr = distr.k, mu = xbar, sigma = sigma.k/sqrt(nstar[j]))
        q2 <- min(1, f.ratio * k.ratio)
        ystar[j] <- ifelse(runif(1) <= q2, y2star, ystar[j])
    }
    return(ystar)
  }
}
\keyword{internal}