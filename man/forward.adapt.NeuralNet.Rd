\encoding{latin1}
\name{forward.adapt.R.NeuralNet}
\alias{forward.adapt.R.NeuralNet}
\alias{forward.adapt.C.NeuralNet}

\title{Perform the forward pass in the adaptative training.}
\description{This function is similar to \code{\link{forwardpass.R.NeuralNet}} but the derivative of the activation function of the Neural Network neurons are also calculated. }

\usage{ forward.adapt.R.NeuralNet(net,Pvector) }

\arguments{
\item{net}{Neural Network to adapt.}
\item{Pvector}{Input vector to propagate.}
}

\value{This function returns the forward passed Neural Network.}
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
\code{\link{forwardpass.R.neuron}}, \code{\link{forward.adapt.R.neuron}}
}

\keyword{neural}

