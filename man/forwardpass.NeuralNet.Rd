\encoding{latin1}
\name{forwardpass.R.NeuralNet}
\alias{forwardpass.R.NeuralNet}
\alias{forwardpass.C.NeuralNet}

\title{Simulate a Neural Network response.}
\description{Performs the simulation of a Neural Network propagating the input signals to the neurons's output. Mainly used by the \code{link{sim.NeuralNet}} function.}

\usage{ forwardpass.R.NeuralNet(net,Pvector) }

\arguments{
\item{net}{Neural Network to simulate.}
\item{Pvector}{Input vector to propagate.}
}

\value{This function returns the propagated Neural Network.}
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
\code{\link{forwardpass.R.neuron}}, \code{\link{forward.adapt.R.NeuralNet}}
}

\keyword{neural}

