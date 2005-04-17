\encoding{latin1}
\name{sim.Forward.MLPneuron}
\alias{sim.Forward.MLPneuron}

\title{Simulate a neuron response from a single input vector.}
\description{Performs the simulation of a neuron propagating the input signals to the neurons's output. Mainly used by the \code{\link{sim.Forward.MLPnet}} function.}

\usage{sim.Forward.MLPneuron(net,ind.MLPneuron) }

\arguments{
\item{net}{Neural Network to simulate.}
\item{ind.MLPneuron}{Index of the neuron to simulate.}
}

\value{This function returns the propagated neuron.}
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
\code{\link{sim.Forward.MLPnet}} }
}

\keyword{neural}

