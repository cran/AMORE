\name{backpropagate.adapt.R.neuron}
\alias{backpropagate.adapt.R.neuron}

\title{Backpropagates the single pattern error modifying accordingly the neuron's weights and bias.}
\description{ This is the original R code version mainly written to be called by \link{backpropagate.adapt.R.NeuralNet}. Since this latter function has been rewritten (\link{backpropagate.adapt.C.NeuralNet}) to call the C code corresponding function (\emph{BackpropagateAdaptNeuronC}), it has been kept just for educational purposes and it is not meant to be used otherwise.}

\usage{ backpropagate.adapt.R.neuron(net, ind.neuron) }

\arguments{
\item{net}{Neural Network to train.}
\item{ind.neuron}{Index of the neuron that needs to suffer the backpropagation effect.}
}

\value{This function returns a neuron object with weights and bias adjusted by the adaptative backpropagation with momentum method.}
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
\code{\link{backpropagate.adapt.R.NeuralNet}}, \code{\link{adapt.R.NeuralNet}}, \code{\link{train}}, \code{\link{train.compare}}
}


\keyword{neural}

