#' Distribution function half Normal
#'
#' Computes the cdf.
#'
#' For internal use
#'
#' @keywords internal
#' @examples
#'
#' ## The function is currently defined as
#' function(q, mean = 0, sd = 1) {
#'   ifelse(q < 0, 0, 1) * (pnorm(q, mean, sd) - pnorm(
#'     0, mean,
#'     sd
#'   )) / (1 - pnorm(0, mean, sd))
#' }
phalfnorm <-
  function(q, mean = 0, sd = 1) {
    ifelse(q < 0, 0, 1) * (pnorm(q, mean, sd) - pnorm(
      0, mean,
      sd
    )) / (1 - pnorm(0, mean, sd))
  }
