\name{select.activation.function}
\alias{select.activation.function}
\title{Provides R code of the selected activation function.}
\description{Provides random values to the neuron weights and bias so as to start with. It is usually called by the random.init.NeuralNet function during the construction of the neural object by the \emph{newff} function. 
}
\usage{ select.activation.function(activation.function)}
\arguments{
\item{activation.function}{ Activation function name. Currently the user may choose amongst \emph{purelin}, \emph{tansig}, \emph{sigmoid}, \emph{hardlim} and \emph{custom}. If \emph{custom} is chosen the the user must manually assign the neuron  \emph{f0} and \emph{f1} functions.}
 }
\value{\emph{select.activation.function} returns a list with two elements. The first, \emph{f0} is the R code selected to serve as the neuron activation function. The second, \emph{f1} is the R code of the activation function derivative.}
\author{
Manuel Castej�n Limas. 	            \email{manuel.castejon@unileon.es}\cr
Joaquin Ordieres Mer�.	            \email{joaquin.ordieres@dim.unirioja.es}\cr
Ana Gonz�lez Marcos.                \email{ana.gonzalez@unileon.es} \cr
Alpha V. Pern�a Espinoza.           \email{alpha.pernia@alum.unirioja.es}\cr
Eliseo P. Vergara Gonz�lez.         \email{eliseo.vergara@dim.unirioja.es}\cr
Francisco Javier Martinez de Pis�n. \email{francisco.martinez@dim.unirioja.es}\cr
Fernando Alba El�as.                \email{fernando.alba@unavarra.es}\cr
}
\seealso{
\code{\link{init.neuron}}, \code{\link{newff}}
}
\keyword{neural}