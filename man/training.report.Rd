\encoding{latin1}
\name{training.report}
\alias{training.report}

\title{Neural network training report generator function.}

\description{Function in charge of ploting graphics and writting messages showing the performance of the neural network.}

\usage{
training.report(net,P,T, epoch, error.criterium)
}
\arguments{
\item{net}{Neural Network to train.}
\item{P}{Training set input values.}
\item{T}{Training set output values}
\item{epoch}{Current epoch.}
\item{error.criterium}{Criterium used to measure the goodness of fit.}
}

\value{This function does not return any value. Just useful for printing and plotting.}
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
Simon Haykin. Neural Networks. A comprehensive foundation. 2nd Edition. \cr \cr
}

\seealso{
\code{\link{train}}, \code{\link{train.compare}}
}

\keyword{neural}

