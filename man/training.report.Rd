\encoding{latin1}
\name{training.report}
\alias{training.report}
\title{Neural network training report generator function.}

\description{Function in charge of reporting the behavior of the network training. The users should modify this function according to their needs.}

\usage{
training.report(net,P,T, idx.show, error.criterium)
}
\arguments{
\item{net}{Neural Network to train.}
\item{P}{Training set input values.}
\item{T}{Training set output values}
\item{idx.show}{Current show index.}
\item{error.criterium}{Criterium used to measure the goodness of fit.}
}

\value{This function does not return any value. Just useful for printing and plotting.}
\author{
Manuel Castej� Limas. 	            \email{manuel.castejon@unileon.es}\cr
Joaquin Ordieres Mer�	            \email{joaquin.ordieres@dim.unirioja.es}\cr
Ana Gonz�ez Marcos.                \email{ana.gonzalez@unileon.es} \cr
Alpha V. Pern� Espinoza.           \email{alpha.pernia@alum.unirioja.es}\cr
Eliseo P. Vergara Gonzalez.         \email{eliseo.vergara@dim.unirioja.es}\cr
Francisco Javier Martinez de Pis�. \email{francisco.martinez@dim.unirioja.es}\cr
Fernando Alba El�s.                \email{fernando.alba@unavarra.es}\cr
}

\references{
Simon Haykin. Neural Networks. A comprehensive foundation. 2nd Edition. \cr \cr
}

\seealso{
\code{\link{train}}
}

\keyword{neural}

