pkgname <- "BNPdensity"
source(file.path(R.home("share"), "R", "examples-header.R"))
options(warn = 1)
options(pager = "console")
library('BNPdensity')

assign(".oldSearch", search(), pos = 'CheckExEnv')
cleanEx()
nameEx("BNPdensity-package")
### * BNPdensity-package

flush(stderr()); flush(stdout())

### Name: BNPdensity-package
### Title: Bayesian nonparametric density estimation
### Aliases: BNPdensity-package BNPdensity
### Keywords: package

### ** Examples

example(MixNRMI1)
example(MixNRMI2)



cleanEx()
nameEx("Enzyme1.out")
### * Enzyme1.out

flush(stderr()); flush(stdout())

### Name: Enzyme1.out
### Title: Fit of MixNRMI1 function to the enzyme dataset
### Aliases: Enzyme1.out
### Keywords: datasets

### ** Examples

data(Enzyme1.out)



cleanEx()
nameEx("Enzyme2.out")
### * Enzyme2.out

flush(stderr()); flush(stdout())

### Name: Enzyme2.out
### Title: Fit of MixNRMI2 function to the enzyme dataset
### Aliases: Enzyme2.out
### Keywords: datasets

### ** Examples

data(Enzyme2.out)



cleanEx()
nameEx("Galaxy1.out")
### * Galaxy1.out

flush(stderr()); flush(stdout())

### Name: Galaxy1.out
### Title: Fit of MixNRMI1 function to the galaxy dataset
### Aliases: Galaxy1.out
### Keywords: datasets

### ** Examples

data(Galaxy1.out)



cleanEx()
nameEx("Galaxy2.out")
### * Galaxy2.out

flush(stderr()); flush(stdout())

### Name: Galaxy2.out
### Title: Fit of MixNRMI2 function to the galaxy dataset
### Aliases: Galaxy2.out
### Keywords: datasets

### ** Examples

data(Galaxy2.out)



cleanEx()
nameEx("MixNRMI1")
### * MixNRMI1

flush(stderr()); flush(stdout())

### Name: MixNRMI1
### Title: Normalized Random Measures Mixture of Type I
### Aliases: MixNRMI1
### Keywords: distribution models nonparametrics

### ** Examples

### Example 1
## Not run: 
##D # Data
##D data(acidity)
##D x <- acidity
##D # Fitting the model under default specifications
##D out <- MixNRMI1(x)
##D # Plotting density estimate + 95##D 
##D attach(out)
##D m <- ncol(qx)
##D ymax <- max(qx[,m])
##D par(mfrow=c(1,1))
##D hist(x,probability=TRUE,breaks=20,col=grey(.9),ylim=c(0,ymax))
##D lines(xx,qx[,1],lwd=2)
##D lines(xx,qx[,2],lty=3,col=4)
##D lines(xx,qx[,m],lty=3,col=4)
##D detach()
## End(Not run)

### Example 2
## Do not run
# set.seed(123456)
# data(enzyme)
# x <- enzyme
# Enzyme1.out <- MixNRMI1(x, Alpha = 1, Beta = 0.007, Gama = 0.5, distr.k = 2,
#                distr.p0 = 2, mu.p0 = 10, sigma.p0 = 10, asigma = 1, bsigma = 1,
#                Nit = 5000, Pbi = 0.2)
# The output of this run is already loaded in the package
# To show results run the following
# Data
data(enzyme)
x <- enzyme
data(Enzyme1.out)
attach(Enzyme1.out)
# Plotting density estimate + 95% credible interval
m <- ncol(qx)
ymax <- max(qx[,m])
par(mfrow=c(1,1))
hist(x,probability=TRUE,breaks=20,col=grey(.9),ylim=c(0,ymax))
lines(xx,qx[,1],lwd=2)
lines(xx,qx[,2],lty=3,col=4)
lines(xx,qx[,m],lty=3,col=4)
# Plotting number of clusters
par(mfrow=c(2,1))
plot(R,type="l",main="Trace of R")
hist(R,breaks=min(R-1):max(R),probability=TRUE)
# Plotting sigma
par(mfrow=c(2,1))
plot(S,type="l",main="Trace of sigma")
hist(S,nclass=20,probability=TRUE,main="Histogram of sigma")
# Plotting u
par(mfrow=c(2,1))
plot(U,type="l",main="Trace of U")
hist(U,nclass=20,probability=TRUE,main="Histogram of U")
# Plotting cpo
par(mfrow=c(2,1))
plot(cpo,main="Scatter plot of CPO's")
boxplot(cpo,horizontal=TRUE,main="Boxplot of CPO's")
print(paste('Average log(CPO)=',round(mean(log(cpo)),4)))
print(paste('Median log(CPO)=',round(median(log(cpo)),4)))
detach()

### Example 3
## Do not run
# set.seed(123456)
# data(galaxy)
# x <- galaxy
# Galaxy1.out <- MixNRMI1(x, Alpha = 1, Beta = 0.015, Gama = 0.5,
#                distr.k = 1, distr.p0 = 2, mu.p0 = 20, sigma.p0 = 20,
#                asigma = 1, bsigma = 1, Nit = 5000, Pbi = 0.2)
# The output of this run is already loaded in the package
# To show results run the following
# Data
data(galaxy)
x <- galaxy
data(Galaxy1.out)
attach(Galaxy1.out)
# Plotting density estimate + 95% credible interval
m <- ncol(qx)
ymax <- max(qx[,m])
par(mfrow=c(1,1))
hist(x,probability=TRUE,breaks=20,col=grey(.9),ylim=c(0,ymax))
lines(xx,qx[,1],lwd=2)
lines(xx,qx[,2],lty=3,col=4)
lines(xx,qx[,m],lty=3,col=4)
# Plotting number of clusters
par(mfrow=c(2,1))
plot(R,type="l",main="Trace of R")
hist(R,breaks=min(R-1):max(R),probability=TRUE)
# Plotting sigma
par(mfrow=c(2,1))
plot(S,type="l",main="Trace of sigma")
hist(S,nclass=20,probability=TRUE,main="Histogram of sigma")
# Plotting u
par(mfrow=c(2,1))
plot(U,type="l",main="Trace of U")
hist(U,nclass=20,probability=TRUE,main="Histogram of U")
# Plotting cpo
par(mfrow=c(2,1))
plot(cpo,main="Scatter plot of CPO's")
boxplot(cpo,horizontal=TRUE,main="Boxplot of CPO's")
print(paste('Average log(CPO)=',round(mean(log(cpo)),4)))
print(paste('Median log(CPO)=',round(median(log(cpo)),4)))
detach()



graphics::par(get("par.postscript", pos = 'CheckExEnv'))
cleanEx()
nameEx("MixNRMI2")
### * MixNRMI2

flush(stderr()); flush(stdout())

### Name: MixNRMI2
### Title: Normalized Random Measures Mixture of Type II
### Aliases: MixNRMI2
### Keywords: distribution models nonparametrics

### ** Examples

## Not run: 
##D ### Example 1
##D # Data
##D data(acidity)
##D x <- acidity
##D # Fitting the model under default specifications
##D out <- MixNRMI2(x)
##D # Plotting density estimate + 95##D 
##D attach(out)
##D m <- ncol(qx)
##D ymax <- max(qx[,m])
##D par(mfrow=c(1,1))
##D hist(x,probability=TRUE,breaks=20,col=grey(.9),ylim=c(0,ymax))
##D lines(xx,qx[,1],lwd=2)
##D lines(xx,qx[,2],lty=3,col=4)
##D lines(xx,qx[,m],lty=3,col=4)
##D detach()
## End(Not run)

### Example 2
## Do not run
# set.seed(123456)
# data(enzyme)
# x <- enzyme
# Enzyme2.out <- MixNRMI2(x, Alpha = 1, Beta = 0.007, Gama = 0.5, distr.k = 2,
#                 distr.py0 = 2, mu.py0 = 10, sigma.py0 = 10,
#                 distr.pz0 = 2, mu.pz0 = 1, sigma.pz0 = 1,
#                 Nit = 5000, Pbi = 0.2)
# The output of this run is already loaded in the package
# To show results run the following
# Data
data(enzyme)
x <- enzyme
data(Enzyme2.out)
attach(Enzyme2.out)
# Plotting density estimate + 95% credible interval
m <- ncol(qx)
ymax <- max(qx[,m])
par(mfrow=c(1,1))
hist(x,probability=TRUE,breaks=20,col=grey(.9),ylim=c(0,ymax))
lines(xx,qx[,1],lwd=2)
lines(xx,qx[,2],lty=3,col=4)
lines(xx,qx[,m],lty=3,col=4)
# Plotting number of clusters
par(mfrow=c(2,1))
plot(R,type="l",main="Trace of R")
hist(R,breaks=min(R-1):max(R),probability=TRUE)
# Plotting u
par(mfrow=c(2,1))
plot(U,type="l",main="Trace of U")
hist(U,nclass=20,probability=TRUE,main="Histogram of U")
# Plotting cpo
par(mfrow=c(2,1))
plot(cpo,main="Scatter plot of CPO's")
boxplot(cpo,horizontal=TRUE,main="Boxplot of CPO's")
print(paste('Average log(CPO)=',round(mean(log(cpo)),4)))
print(paste('Median log(CPO)=',round(median(log(cpo)),4)))
detach()

### Example 3
## Do not run
# set.seed(123456)
# data(galaxy)
# x <- galaxy
# Galaxy2.out <- MixNRMI2(x, Alpha = 1, Beta = 0.015, Gama = 0.5, distr.k = 1,
#                 distr.py0 = 2, mu.py0 = 20, sigma.py0 = 20,
#                 distr.pz0 = 2, mu.pz0 = 1, sigma.pz0 = 1,
#                 Nit = 5000, Pbi = 0.2)
# The output of this run is already loaded in the package
# To show results run the following
# Data
data(galaxy)
x <- galaxy
data(Galaxy2.out)
attach(Galaxy2.out)
# Plotting density estimate + 95% credible interval
m <- ncol(qx)
ymax <- max(qx[,m])
par(mfrow=c(1,1))
hist(x,probability=TRUE,breaks=20,col=grey(.9),ylim=c(0,ymax))
lines(xx,qx[,1],lwd=2)
lines(xx,qx[,2],lty=3,col=4)
lines(xx,qx[,m],lty=3,col=4)
# Plotting number of clusters
par(mfrow=c(2,1))
plot(R,type="l",main="Trace of R")
hist(R,breaks=min(R-1):max(R),probability=TRUE)
# Plotting u
par(mfrow=c(2,1))
plot(U,type="l",main="Trace of U")
hist(U,nclass=20,probability=TRUE,main="Histogram of U")
# Plotting cpo
par(mfrow=c(2,1))
plot(cpo,main="Scatter plot of CPO's")
boxplot(cpo,horizontal=TRUE,main="Boxplot of CPO's")
print(paste('Average log(CPO)=',round(mean(log(cpo)),4)))
print(paste('Median log(CPO)=',round(median(log(cpo)),4)))
detach()



graphics::par(get("par.postscript", pos = 'CheckExEnv'))
cleanEx()
nameEx("Mv")
### * Mv

flush(stderr()); flush(stdout())

### Name: Mv
### Title: Continuous Jump heights function
### Aliases: Mv
### Keywords: internal

### ** Examples

## The function is currently defined as
function (u = 0.5, alpha = 1, beta = 1, gama = 1/2, low = 1e-04, 
    upp = 10, N = 5001) 
{
    x <- -log(seq(from = exp(-low), to = exp(-upp), length = N))
    f <- alpha/gamma(1 - gama) * x^(-(1 + gama)) * exp(-(u + 
        beta) * x)
    dx <- diff(x)
    h <- (f[-1] + f[-N])/2
    Mv <- rep(0, N)
    for (i in seq(N - 1, 1)) Mv[i] <- Mv[i + 1] + dx[i] * h[i]
    return(list(v = x, Mv = Mv))
  }



cleanEx()
nameEx("MvInv")
### * MvInv

flush(stderr()); flush(stdout())

### Name: MvInv
### Title: Inversed jump heights function
### Aliases: MvInv
### Keywords: internal

### ** Examples

## The function is currently defined as
function (w, u = 0.5, alpha = 1, beta = 1, gama = 1/2, N = 3001) 
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



cleanEx()
nameEx("acidity")
### * acidity

flush(stderr()); flush(stdout())

### Name: acidity
### Title: Acidity Index Dataset
### Aliases: acidity
### Keywords: datasets

### ** Examples

data(acidity)
hist(acidity)



cleanEx()
nameEx("comp1")
### * comp1

flush(stderr()); flush(stdout())

### Name: comp1
### Title: Ties function: univariate
### Aliases: comp1
### Keywords: internal

### ** Examples

## The function is currently defined as
function (y) 
{
    n <- length(y)
    mat <- outer(y, y, "==")
    jstar <- led <- rep(FALSE, n)
    for (j in seq(n)) {
        if (!led[j]) {
            jstar[j] <- TRUE
            if (j == n) 
                break
            ji <- seq(j + 1, n)
            tt <- mat[ji, j] %in% TRUE
            led[ji] <- led[ji] | tt
        }
        if (all(led[-seq(j)])) 
            break
    }
    ystar <- y[jstar]
    nstar <- apply(mat[, jstar], 2, sum)
    r <- length(nstar)
    idx <- match(y, ystar)
    return(list(ystar = ystar, nstar = nstar, r = r, idx = idx))
  }



cleanEx()
nameEx("comp2")
### * comp2

flush(stderr()); flush(stdout())

### Name: comp2
### Title: Ties function: bivariate
### Aliases: comp2
### Keywords: internal

### ** Examples

## The function is currently defined as
function (y, z) 
{
    if (length(y) != length(z)) 
        stop("Vectors y and z should have equal length!")
    n <- length(y)
    matY <- outer(y, y, "==")
    matZ <- outer(z, z, "==")
    mat <- matY & matZ
    jstar <- led <- rep(FALSE, n)
    for (j in seq(n)) {
        if (!led[j]) {
            jstar[j] <- TRUE
            if (j == n) 
                break
            ji <- seq(j + 1, n)
            tt <- mat[ji, j] %in% TRUE
            led[ji] <- led[ji] | tt
        }
        if (all(led[-seq(j)])) 
            break
    }
    ystar <- y[jstar]
    zstar <- z[jstar]
    nstar <- apply(mat[, jstar], 2, sum)
    rstar <- length(nstar)
    idx <- match(y, ystar)
    return(list(ystar = ystar, zstar = zstar, nstar = nstar, 
        rstar = rstar, idx = idx))
  }



cleanEx()
nameEx("cpo")
### * cpo

flush(stderr()); flush(stdout())

### Name: cpo
### Title: Conditional predictive ordinate function
### Aliases: cpo
### Keywords: internal

### ** Examples

## The function is currently defined as
function (obj) 
{
    fx <- obj$fx
    cpo <- 1/apply(1/fx, 1, mean)
    return(cpo)
  }



cleanEx()
nameEx("dk")
### * dk

flush(stderr()); flush(stdout())

### Name: dk
### Title: Kernel density function
### Aliases: dk
### Keywords: internal

### ** Examples

## The function is currently defined as
function (x, distr = NULL, mu = NULL, sigma = NULL) 
{
    if (is.null(distr)) {
        stop("Argument \"distr\" should be defined numeric with possible values 1,2,3,4 or 5")
    }
    else if (distr == 1) {
        a <- ifelse(is.null(mu), 0, mu)
        b <- ifelse(is.null(sigma), 1, sigma)
        dk <- dnorm(x, mean = a, sd = b)
    }
    else if (distr == 2) {
        a <- ifelse(is.null(mu), 0, mu)
        b <- ifelse(is.null(sigma), 1/sqrt(2), sigma/sqrt(2))
        dk <- exp(-abs(x - a)/b)/(2 * b)
    }
    else if (distr == 3) {
        a <- ifelse(is.null(mu), exp(1/2), log(mu/sqrt(1 + (sigma/mu)^2)))
        b <- ifelse(is.null(sigma), exp(1) * (exp(1) - 1), sqrt(log(1 + 
            (sigma/y)^2)))
        dk <- dlnorm(x, meanlog = a, sdlog = b)
    }
    else if (distr == 4) {
        a <- ifelse(is.null(mu), 1, mu^2/sigma^2)
        b <- ifelse(is.null(sigma), 1, mu/sigma^2)
        dk <- dgamma(x, shape = a, rate = b)
    }
    else if (distr == 5) {
        a <- ifelse(is.null(mu), 0.5, (1 - mu) * (mu/sigma)^2 - 
            mu)
        b <- ifelse(is.null(sigma), 1/sqrt(12), (mu * (1 - mu)/sigma^2 - 
            1) * (1 - mu))
        if (any(c(a, b) <= 0)) 
            stop(paste("\nNegative Beta parameters:\n a =", a, 
                ";\t b =", b))
        dk <- dbeta(x, shape1 = a, shape2 = b)
    }
    else {
        stop("Argument \"distr\" should be defined numeric with possible values 1,2,3,4 or 5")
    }
    return(dk)
  }



cleanEx()
nameEx("enzyme")
### * enzyme

flush(stderr()); flush(stdout())

### Name: enzyme
### Title: Enzyme Dataset
### Aliases: enzyme
### Keywords: datasets

### ** Examples

data(enzyme)
hist(enzyme)



cleanEx()
nameEx("fcondXA")
### * fcondXA

flush(stderr()); flush(stdout())

### Name: fcondXA
### Title: Conditional density evaluation in the semiparametric model
### Aliases: fcondXA
### Keywords: internal

### ** Examples

## The function is currently defined as
function (x, distr = 1, Tau, J, sigma) 
{
    pJ <- J/sum(J)
    K <- matrix(NA, nrow = length(Tau), ncol = length(x))
    for (i in seq(Tau)) {
        K[i, ] <- dk(x, distr = distr, mu = Tau[i], sigma = sigma)
    }
    fcondXA <- apply(K, 2, function(x) sum(x * pJ))
    return(fcondXA)
  }



cleanEx()
nameEx("fcondXA2")
### * fcondXA2

flush(stderr()); flush(stdout())

### Name: fcondXA2
### Title: Conditional density evaluation in the fully nonparametric model
### Aliases: fcondXA2
### Keywords: internal

### ** Examples

## The function is currently defined as
function (x, distr = 1, Tauy, Tauz, J) 
{
    pJ <- J/sum(J)
    K <- matrix(NA, nrow = length(Tauy), ncol = length(x))
    for (i in seq(Tauy)) {
        K[i, ] <- dk(x, distr = distr, mu = Tauy[i], sigma = Tauz[i])
    }
    fcondXA2 <- apply(K, 2, function(x) sum(x * pJ))
    return(fcondXA2)
  }



cleanEx()
nameEx("fcondYXA")
### * fcondYXA

flush(stderr()); flush(stdout())

### Name: fcondYXA
### Title: Conditional posterior distribution of the latents Y
### Aliases: fcondYXA
### Keywords: internal

### ** Examples

## The function is currently defined as
function (x, distr = 1, Tau, J, sigma) 
{
    K <- matrix(NA, nrow = length(Tau), ncol = length(x))
    for (i in seq(Tau)) {
        K[i, ] <- dk(x, distr = distr, mu = Tau[i], sigma = sigma) * 
            J[i]
    }
    pK <- prop.table(K, margin = 2)
    y <- apply(pK, 2, function(x) sample(Tau, size = 1, prob = x))
    return(y)
  }



cleanEx()
nameEx("fcondYZXA")
### * fcondYZXA

flush(stderr()); flush(stdout())

### Name: fcondYZXA
### Title: Conditional posterior distribution of the bivariate latents
###   (Y,Z)
### Aliases: fcondYZXA
### Keywords: internal

### ** Examples

## The function is currently defined as
function (x, distr = 1, Tauy, Tauz, J) 
{
    K <- matrix(NA, nrow = length(Tauy), ncol = length(x))
    for (i in seq(Tauy)) {
        K[i, ] <- dk(x, distr = distr, mu = Tauy[i], sigma = Tauz[i]) * 
            J[i]
    }
    if (any(is.na(K))) 
        print(K, Tauy, Tauz, J)
    pK <- prop.table(K, margin = 2)
    j <- apply(pK, 2, function(x) sample(length(Tauy), size = 1, 
        prob = x))
    return(matrix(c(y = Tauy[j], z = Tauz[j]), nrow = length(x), 
        ncol = 2))
  }



cleanEx()
nameEx("galaxy")
### * galaxy

flush(stderr()); flush(stdout())

### Name: galaxy
### Title: Galaxy Data Set
### Aliases: galaxy
### Keywords: datasets

### ** Examples

data(galaxy)
hist(galaxy)



cleanEx()
nameEx("gs3")
### * gs3

flush(stderr()); flush(stdout())

### Name: gs3
### Title: Conditional posterior distribution of latent U
### Aliases: gs3
### Keywords: internal

### ** Examples

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



cleanEx()
nameEx("gs4")
### * gs4

flush(stderr()); flush(stdout())

### Name: gs4
### Title: Resampling Ystar function
### Aliases: gs4
### Keywords: internal

### ** Examples

## The function is currently defined as
function (ystar, x, idx, distr.k, sigma.k, distr.p0, mu.p0, sigma.p0) 
{
    r <- length(ystar)
    nstar <- as.numeric(table(idx))
    for (j in seq(r)) {
        id <- which(!is.na(match(idx, j)))
        xj <- x[id]
        xbar <- sum(xj)/nstar[j]
        y2star <- rk(1, distr = distr.k, mu = xbar, sigma = sigma.k/sqrt(nstar[j]))
        f.ratio <- rfystar(y2star, ystar[j], xj, distr = distr.k, sigma = sigma.k, 
            distr.p0 = distr.p0, mu.p0 = mu.p0, sigma.p0 = sigma.p0)
        k.ratio <- dk(ystar[j], distr = distr.k, mu = xbar, sigma = sigma.k/sqrt(nstar[j]))/dk(y2star, 
            distr = distr.k, mu = xbar, sigma = sigma.k/sqrt(nstar[j]))
        q2 <- min(1, f.ratio * k.ratio)
        ystar[j] <- ifelse(runif(1) <= q2, y2star, ystar[j])
    }
    return(ystar)
  }



cleanEx()
nameEx("gs5")
### * gs5

flush(stderr()); flush(stdout())

### Name: gs5
### Title: Conditional posterior distribution of sigma
### Aliases: gs5
### Keywords: internal

### ** Examples

## The function is currently defined as
function (sigma, x, y, distr = 1, asigma = 1, bsigma = 2, delta = 4) 
{
    sigmaStar <- rgamma(1, shape = delta, rate = delta/sigma)
    sigmaT <- sigma
    qgammas <- sigmaT/sigmaStar
    Qgammas <- sigmaStar/sigmaT
    Term2 <- qgammas^(2 * delta - 1) * exp(-delta * (qgammas - 
        Qgammas))
    Kgamma <- Qgammas^(asigma - 1) * exp(-bsigma * (sigmaStar - 
        sigmaT))
    Prod <- 1
    for (i in seq(length(x))) {
        Prod <- Prod * (dk(x[i], distr = distr, mu = y[i], sigma = sigmaStar)/dk(x[i], 
            distr = distr, mu = y[i], sigma = sigmaT))
    }
    q3 <- min(1, Kgamma * Prod * Term2)
    sigma <- ifelse(runif(1) <= q3, sigmaStar, sigmaT)
    return(sigma)
  }



cleanEx()
nameEx("gsYZstar")
### * gsYZstar

flush(stderr()); flush(stdout())

### Name: gsYZstar
### Title: Resampling Ystar and Zstar function
### Aliases: gsYZstar
### Keywords: internal

### ** Examples

## The function is currently defined as
function (ystar, zstar, nstar, rstar, idx, x, delta, distr.k, 
    distr.py0, mu.py0, sigma.py0, distr.pz0, mu.pz0, sigma.pz0) 
{
    for (j in seq(rstar)) {
        id <- which(!is.na(match(idx, j)))
        xj <- x[id]
        xbar <- sum(xj)/nstar[j]
        y2star <- rk(1, distr = distr.k, mu = xbar, sigma = 1/sqrt(nstar[j]))
        f.ratio <- rfystar(y2star, ystar[j], xj, distr = distr.k, sigma.k = zstar[j], 
            distr.p0 = distr.py0, mu.p0 = mu.py0, sigma.p0 = sigma.py0)
        k.ratioNum <- dk(ystar[j], distr = distr.k, mu = xbar, 
            sigma = 1/sqrt(nstar[j]))
        k.ratioDen <- dk(y2star, distr = distr.k, mu = xbar, 
            sigma = 1/sqrt(nstar[j]))
        k.ratio <- k.ratioNum/k.ratioDen
        q2 <- min(1, f.ratio * k.ratio)
        if (runif(1) <= q2) 
            ystar[j] <- y2star
        z2star <- rk(1, distr = distr.pz0, mu = zstar[j], sigma = zstar[j]/sqrt(delta))
        f.ratio <- rfzstar(ystar[j], xj, distr = distr.k, sigma.k = z2star, 
	      sigma.k2 = zstar[j], distr.p0 = distr.pz0, mu.p0 = mu.pz0, 
	      sigma.p0 = sigma.pz0)
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



cleanEx()
nameEx("p0")
### * p0

flush(stderr()); flush(stdout())

### Name: p0
### Title: Centering function
### Aliases: p0
### Keywords: internal

### ** Examples

## The function is currently defined as
function (x, distr = NULL, mu = NULL, sigma = NULL) 
{
    if (is.null(distr)) {
        stop("Argument \"distr\" should be defined numeric with possible values 1,2, or 3")
    }
    else if (distr == 1) {
        a <- ifelse(is.null(mu), 0, mu)
        b <- ifelse(is.null(sigma), 1, sigma)
        p0 <- dnorm(x, mean = a, sd = b)
    }
    else if (distr == 2) {
        a <- ifelse(is.null(mu), 1, mu^2/sigma^2)
        b <- ifelse(is.null(sigma), 1, mu/sigma^2)
        p0 <- dgamma(x, shape = a, rate = b)
    }
    else if (distr == 3) {
        a <- ifelse(is.null(mu), 0.5, (1 - mu) * (mu/sigma)^2 - 
            mu)
        b <- ifelse(is.null(sigma), 1/sqrt(12), (mu * (1 - mu)/sigma^2 - 
            1) * (1 - mu))
        if (any(c(a, b) <= 0)) 
            stop(paste("\nNegative Beta parameters:\n a =", a, 
                ";\t b =", b))
        p0 <- dbeta(x, shape1 = a, shape2 = b)
    }
    else {
        stop("Argument \"distr\" should be defined numeric with possible values 1,2, or 3")
    }
    return(p0)
  }



cleanEx()
nameEx("rfystar")
### * rfystar

flush(stderr()); flush(stdout())

### Name: fystar
### Title: Conditional posterior distribution of the distict Ystar
### Aliases: fystar
### Keywords: internal

### ** Examples

## The function is currently defined as
function (v, v2, x, distr.k, sigma.k, distr.p0, mu.p0, sigma.p0) 
{
    alpha <- p0(v, distr = distr.p0, mu = mu.p0, sigma = sigma.p0)/
             p0(v2, distr = distr.p0, mu = mu.p0, sigma = sigma.p0)
    Prod <- 1
    for (i in seq(length(x))) {
        fac <- dk(x[i], distr = distr.k, mu = v, sigma = sigma.k)/
               dk(x[i], distr = distr.k, mu = v2, sigma = sigma.k)
        Prod <- Prod * fac
    }
    f <- alpha * Prod
    return(f)
  }



cleanEx()
nameEx("rfzstar")
### * rfzstar

flush(stderr()); flush(stdout())

### Name: rfzstar
### Title: Conditional posterior distribution of the distict Zstar
### Aliases: rfzstar
### Keywords: internal

### ** Examples

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



cleanEx()
nameEx("rk")
### * rk

flush(stderr()); flush(stdout())

### Name: rk
### Title: Kernel density sampling function
### Aliases: rk
### Keywords: internal

### ** Examples

## The function is currently defined as
function (n, distr = NULL, mu = NULL, sigma = NULL) 
{
    if (is.null(distr)) {
        stop("Argument \"distr\" should be defined numeric with possible values 1,2,3,4 or 5")
    }
    else if (distr == 1) {
        a <- ifelse(is.null(mu), 0, mu)
        b <- ifelse(is.null(sigma), 1, sigma)
        rk <- rnorm(n, mean = a, sd = b)
    }
    else if (distr == 2) {
        a <- ifelse(is.null(mu), 0, mu)
        b <- ifelse(is.null(sigma), 1/sqrt(2), sigma/sqrt(2))
        rk <- a + b * sample(c(-1, +1), size = n, replace = TRUE) * 
            rexp(n)
    }
    else if (distr == 3) {
        a <- ifelse(is.null(mu), exp(1/2), log(mu/sqrt(1 + (sigma/mu)^2)))
        b <- ifelse(is.null(sigma), exp(1) * (exp(1) - 1), sqrt(log(1 + 
            (sigma/y)^2)))
        rk <- rlnorm(n, meanlog = a, sdlog = b)
    }
    else if (distr == 4) {
        a <- ifelse(is.null(mu), 1, mu^2/sigma^2)
        b <- ifelse(is.null(sigma), 1, mu/sigma^2)
        rk <- rgamma(n, shape = a, rate = b)
    }
    else if (distr == 5) {
        a <- ifelse(is.null(mu), 0.5, (1 - mu) * (mu/sigma)^2 - 
            mu)
        b <- ifelse(is.null(sigma), 1/sqrt(12), (mu * (1 - mu)/sigma^2 - 
            1) * (1 - mu))
        if (any(c(a, b) <= 0)) 
            stop(paste("\nNegative Beta parameters:\n a =", a, 
                ";\t b =", b))
        rk <- rbeta(n, shape1 = a, shape2 = b)
    }
    else {
        stop("Argument \"distr\" should be defined numeric with possible values 1,2,3,4 or 5")
    }
    return(rk)
  }



### * <FOOTER>
###
cat("Time elapsed: ", proc.time() - get("ptime", pos = 'CheckExEnv'),"\n")
grDevices::dev.off()
###
### Local variables: ***
### mode: outline-minor ***
### outline-regexp: "\\(> \\)?### [*]+" ***
### End: ***
quit('no')
