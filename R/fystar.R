fystar <-
function (v, x, distr.k, sigma.k, distr.p0, mu.p0, sigma.p0) 
{
    alpha <- p0(v, distr = distr.p0, mu = mu.p0, sigma = sigma.p0)
    Prod <- 1
    for (i in seq(length(x))) {
        fac <- dk(x[i], distr = distr.k, mu = v, sigma = sigma.k)
        Prod <- Prod * fac
    }
    f <- alpha * Prod
    return(f)
}
