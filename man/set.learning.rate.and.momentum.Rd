\encoding{latin1}
\name{set.learning.rate.and.momentum}
\alias{set.learning.rate.and.momentum}
\title{Set Learning rate and momentum.}
\description{This function sets the learning rate and momentum of every neuron in a given neural network.}
\usage{
set.learning.rate.and.momentum(net, learning.rate, momentum)
}

\arguments{
\item{net}{Neural Network to train.}
\item{learning.rate}{Learning rate to be set in every neuron.}
\item{momentum}{Momentum to be set on every neuron.}
}

\value{This function returns a neural network object with the established values of the learning rate and momentum in every neuron.}
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
