\name{forward.adapt.R.neuron}
\alias{forward.adapt.R.neuron}

\title{Perform the neuron forward pass in the adaptative training.}
\description{This function is similar to \code{\link{forwardpass.R.neuron}} but the derivative of the activation function of the neuron is also calculated. }

\usage{ forward.adapt.R.neuron(net,ind.neuron) }

\arguments{
\item{net}{Neural Network to adapt.}
\item{ind.neuron}{Index of the neuron to forward pass.}
}

\value{This function returns the forward passed neuron.}
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
\code{\link{forwardpass.R.NeuralNet}}, \code{\link{forward.adapt.R.NeuralNet}}
}

\keyword{neural}

