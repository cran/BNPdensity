MvInv <-
function (w, u, alpha, beta, gama, N) 
{
    n <- length(w)
    v <- rep(NA, n)
    x <- -log(seq(from = exp(-1e-05), to = exp(-10), length = N))
    f <- alpha/gamma(1 - gama) * x^(-(1 + gama)) * exp(-(u + 
        beta) * x)
    dx <- diff(x)
    h <- (f[-1] + f[-N])/2
    Mv <- rep(0, N)
    for (i in seq(N - 1, 1)) Mv[i] <- Mv[i + 1] + dx[i] * h[i]
    for (j in seq(n)) v[j] <- x[which.min(Mv > w[j])]
    return(v)
}