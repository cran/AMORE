\name{random.init.neuron}
\alias{random.init.neuron}
\title{Initialize the neuron with random weigths and bias.}
\description{Provides random values to the neuron weights and bias so as to start with. It is usually called by the random.init.NeuralNet function during the construction of the neural object by the \emph{newff} function. 
}
\details{The values are assigned according to the suggestions of \cite{Haykin}.}
\usage{random.init.neuron(net.number.weights, neuron) }
\arguments{
\item{net.number.weights}{Number of bias and weight parameters of the neural network the neuron belongs to.}
\item{neuron}{The neuron object.}
 }
\value{\emph{random.init.neuron} returns the input neuron with bias and weights changed randomly.}
\author{
Manuel Castejón Limas. 	            \email{manuel.castejon@unileon.es}\cr
Joaquin Ordieres Meré.	            \email{joaquin.ordieres@dim.unirioja.es}\cr
Ana González Marcos.                \email{ana.gonzalez@unileon.es} \cr
Alpha V. Pernía Espinoza.           \email{alpha.pernia@alum.unirioja.es}\cr
Eliseo P. Vergara González.         \email{eliseo.vergara@dim.unirioja.es}\cr
Francisco Javier Martinez de Pisón. \email{francisco.martinez@dim.unirioja.es}\cr
Fernando Alba Elías.                \email{fernando.alba@unavarra.es}\cr
}
\seealso{
\code{\link{random.init.NeuralNet}}, \code{\link{init.neuron}}, \code{\link{newff}}
}
\references{
Simon Haykin. Neural Networks. A comprehensive foundation. 2nd Edition, pages 182, 183 and 184. \cr \cr
}
\keyword{neural}
