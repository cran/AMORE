\name{sim.C.NeuralNet}
\alias{sim.NeuralNet}
\alias{sim.C.NeuralNet}
\alias{sim.R.NeuralNet}

\title{Performs the simulation of a neural network providing the output values.}
\description{This function calculates the output values of the neural network for a given data set. Various versions are provided according to different degrees of C code conversion. The \emph{sim.NeuralNet} function is the latest and quickest.}

\usage{
sim.NeuralNet(net,P)
sim.R.NeuralNet(net,P)
sim.C.NeuralNet(net,P)
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

