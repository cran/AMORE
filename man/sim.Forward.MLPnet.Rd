\encoding{latin1}
\name{sim.Forward.MLPnet}
\alias{sim.Forward.MLPnet}
\alias{sim.Forward.MLPnet.R}

\title{Simulate a Neural Network response from a single input vector.}
\description{Performs the simulation of a Neural Network propagating the single input vector to the neurons's output. Mainly used by the \code{\link{sim.MLPnet}} function.}

\usage{
sim.Forward.MLPnet(net,Pvector)
sim.Forward.MLPnet.R(net,Pvector)
}

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
\code{\link{sim.MLPnet}}, \code{\link{sim.Forward.MLPneuron}}}
}

\keyword{neural}

