gsYZstar <-
function (ystar, zstar, nstar, rstar, idx, x, delta, distr.k, 
    distr.py0, mu.py0, sigma.py0, distr.pz0, mu.pz0, sigma.pz0) 
{
    for (j in seq(rstar)) {
        id <- which(!is.na(match(idx, j)))
        xj <- x[id]
        xbar <- sum(xj)/nstar[j]
        y2star <- rk(1, distr = distr.k, mu = xbar, sigma = 1/sqrt(nstar[j]))
        f.ratioNum <- fystar(y2star, xj, distr = distr.k, sigma.k = zstar[j], 
            distr.p0 = distr.py0, mu.p0 = mu.py0, sigma.p0 = sigma.py0)
        f.ratioDen <- fystar(ystar[j], xj, distr = distr.k, sigma.k = zstar[j], 
            distr.p0 = distr.py0, mu.p0 = mu.py0, sigma.p0 = sigma.py0)
        f.ratio <- f.ratioNum/f.ratioDen
        k.ratioNum <- dk(ystar[j], distr = distr.k, mu = xbar, 
            sigma = 1/sqrt(nstar[j]))
        k.ratioDen <- dk(y2star, distr = distr.k, mu = xbar, 
            sigma = 1/sqrt(nstar[j]))
        k.ratio <- k.ratioNum/k.ratioDen
        q2 <- min(1, f.ratio * k.ratio)
        if (runif(1) <= q2) 
            ystar[j] <- y2star
        z2star <- rk(1, distr = distr.pz0, mu = zstar[j], sigma = zstar[j]/sqrt(delta))
        f.ratioNum <- fzstar(ystar[j], xj, distr = distr.k, sigma.k = z2star, 
            distr.p0 = distr.pz0, mu.p0 = mu.pz0, sigma.p0 = sigma.pz0)
        f.ratioDen <- fzstar(ystar[j], xj, distr = distr.k, sigma.k = zstar[j], 
            distr.p0 = distr.pz0, mu.p0 = mu.pz0, sigma.p0 = sigma.pz0)
        f.ratio <- f.ratioNum/f.ratioDen
        k.ratioNum <- dk(zstar[j], distr = distr.pz0, mu = z2star, 
            sigma = z2star/sqrt(delta))
        k.ratioDen <- dk(z2star, distr = distr.pz0, mu = zstar[j], 
            sigma = zstar[j]/sqrt(delta))
        k.ratio <- k.ratioNum/k.ratioDen
        q3 <- min(1, f.ratio * k.ratio)
        if (runif(1) <= q3) 
            zstar[j] <- z2star
    }
    return(list(ystar = ystar, zstar = zstar))
}
