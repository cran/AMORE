\name{train.compare}
\alias{train.compare}

\title{Trains the same neural network according to different error criteria.}
\description{This function trains a neural network according to different error criteria so as to compare the different behaviours.}

\usage{
train.compare(net.start, P, T, ideal=NA, max.epoch, show.step, Stao=1000, criteria=c("MSE","LMLS","TAO")) 
}

\arguments{
\item{net.start}{Neural Network to train.}
\item{P}{Training set input values.}
\item{T}{Training set output values}
\item{ideal}{Clean training set output values. Useful for training robust networks with noisy data.}
\item{max.epoch}{Number of epochs to train.}
\item{show.step}{A report is provided every show.step epochs.}
\item{Stao}{Initial value of the S parameter used by the TAO algorithm.}
\item{criteria}{A vector specifying which criteria should be used.}
}

\value{This function returns a list containing the trained Neural Network objects according to the specified criteria.}
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
Pernia Espinoza, A.V. TAO-ROBUST BACKPROPAGATION LEARNING ALGORITHM \cr \cr
Simon Haykin. Neural Networks. A comprehensive foundation. 2nd Edition. \cr \cr
}

\seealso{
\code{\link{train}}
}

\keyword{neural}

