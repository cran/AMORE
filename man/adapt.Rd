\name{adapt}
\alias{adapt.NeuralNet}
\alias{adapt.R.NeuralNet}
\alias{adapt.C1.NeuralNet}
\alias{adapt.C2.NeuralNet}
\alias{adapt.C3.NeuralNet}
\title{Adaptative on-line single pattern training}
\description{Modifies the neural network weights and biases according to one pattern chosen from the training set. Mostly written to be called by the \link{train} function so as to be applied to the whole training set. 

Currently the user may choose amongst several versions of the same function. The differences lay in the incresing degree of C code programming involve. The slower adapt.R.NeuralNet was the original R code function and adapt.C4.NeuralNet (a.k.a adapt.NeuralNet ) is the more C conversed function and the one that provides higher speeds of execution.}
\usage{
adapt.NeuralNet(net,Pvector,target)
adapt.R.NeuralNet(net,Pvector,target)
adapt.C1.NeuralNet(net,Pvector,target)
adapt.C2.NeuralNet(net,Pvector,target)
adapt.C3.NeuralNet(net,Pvector,target)
}

\arguments{
\item{net}{Neural Network to train.}
\item{Pvector}{Input values of the training pattern.}
\item{target}{Output values of the training pattern.}
}

\value{These functions return a neural network object modified according to the chosen pattern vector.}
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
\code{\link{train}}, \code{\link{train.compare}}
}


\keyword{neural}
