\encoding{latin1}
\name{error.TAO}
\alias{hfun}
\alias{phifun}
\alias{dphifun}

\title{TAO robust error criterium auxiliar functions.}

\description{Auxiliar functions. Not meant to be called from the user but from the \code{\link{error.TAO}} and the \code{\link{deltaE.TAO}} functions.
}

\usage{
hfun(v,k)
phifun(v,k)
dphifun(v,k)
}
\arguments{
\item{v}{Input value.}
\item{k}{Threshold limit.}
}

\value{These functions return a numeric array with dimension equal to the dimension of v.}
\author{
Manuel Castejón Limas. 	            \email{manuel.castejon@unileon.es}\cr
Joaquin Ordieres Meré.	            \email{joaquin.ordieres@dim.unirioja.es}\cr
Ana González Marcos.                \email{ana.gonzalez@unileon.es} \cr
Alpha V. Pernía Espinoza.           \email{alpha.pernia@alum.unirioja.es}\cr
Eliseo P. Vergara Gonzalez.         \email{eliseo.vergara@dim.unirioja.es}\cr
Francisco Javier Martinez de Pisón. \email{francisco.martinez@dim.unirioja.es}\cr
Fernando Alba Elías.                \email{fernando.alba@unavarra.es}\cr
}

\references{
Pernia Espinoza, A.V. TAO-robust backpropagation learning algorithm. Neural Networks. In press. \cr \cr
Simon Haykin. Neural Networks. A comprehensive foundation. 2nd Edition. \cr \cr
}

\seealso{
\code{\link{train}}, \code{\link{train.compare}}
}

\keyword{neural}

