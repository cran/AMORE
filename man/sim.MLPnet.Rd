\encoding{latin1}
\name{sim.MLPnet}
\alias{sim.MLPnet}
\alias{sim.MLPnet.R}
\alias{sim.MLPnet.C}

\title{Performs the simulation of a neural network from an input data set.}
\description{This function calculates the output values of the neural network for a given data set. Various versions are provided according to different degrees of C code conversion. The \emph{sim.MLPnet} function is the latest and quickest.}

\usage{
sim.MLPnet(net,P)
sim.MLPnet.R(net,P)
sim.MLPnet.C(net,P)
}

\arguments{
\item{net}{Neural Network to simulate.}
\item{P}{Data Set input values.}
}

\value{This function returns a matrix containing the output values of the neural network for the given data set.}
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
\code{\link{train}}
}

\keyword{neural}

