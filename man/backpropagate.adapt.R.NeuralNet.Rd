\encoding{latin1}
\name{backpropagate.adapt.R.NeuralNet}
\alias{backpropagate.adapt.R.NeuralNet}
\alias{backpropagate.adapt.C.NeuralNet}


\title{Backpropagates the single pattern error modifying accordingly the Neural Network's weights and biases.}
\description{ Mainly written to be called by the adapt.R.NeuralNet function. Currently two versions are available: the original R code function (backpropagate.adapt.R.NeuralNet) has kept for educational purposses while the C code version (\link{backpropagate.adapt.C.NeuralNet}) provides higher speeds of executions and thus is most likely the one to be preferred.}

\usage{
backpropagate.adapt.R.NeuralNet(net,target)
backpropagate.adapt.C.NeuralNet(net,target)
}

\arguments{
\item{net}{Neural Network to train.}
\item{target}{Training output vector.}
}

\value{This function returns a Neural Network object with weights and biases adjusted by the adaptative backpropagation with momentum method.}
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

