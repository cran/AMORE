\name{newff}
\alias{newff}
\title{ Feedforward Neural Network}
\description{Creates a feedforward artificial neural network according to the structure established by the AMORE package standard.}
\usage{newff(n.inputs, n.hidden, n.outputs, learning.rate.global, momentum.global, error.criterium, Stao, hidden.layer, output.layer) }
\arguments{
\item{n.inputs}{Number of input neurons or predictors.}
\item{n.hidden}{Number of hidden layer neurons.}
\item{n.outputs}{Number of output layer neurons.}
\item{learning.rate.global}{Learning rate.}
\item{momentum.global}{Momentum (Set to 0 if you do not want to use it).}
\item{error.criterium}{ Criterium used to measure to proximity of the neural network prediction to its target. Currently we can choose amongst: 
\itemize{
\item "MSE": Mean Squared Error
\item "LMLS": Least Mean Logarithm Squared (Liano 1996).
\item "TAO": TAO Error (Pernia, 2004).
}}
\item{Stao}{Stao parameter for the TAO error criterium. Unused by the rest of criteria.}
\item{hidden.layer}{Activation function of the hidden layer neurons. Available functions are:
\itemize{
\item "purelin".
\item "tansig". 
\item "sigmoid".
\item "hardlim".
}}
\item{output.layer}{Activation function of the hidden layer neurons according to the former list shown above.}
}}
\value{\emph{newff} returns a feedforward neural network object.}
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
Pernia Espinoza, A.V. TAO-ROBUST BACKPROPAGATION LEARNING ALGORITHM \cr \cr
Simon Haykin. Neural Networks. A comprehensive foundation. 2nd Edition. \cr \cr
}

\seealso{
\code{\link{init.neuron}}, \code{\link{random.init.NeuralNet}}, \code{\link{random.init.neuron}}, \code{\link{select.activation.function}} , \code{\link{init.neuron}}
}

\examples{
#Example 1

library(AMORE)
# P is the input vector
P <- matrix(sample(seq(-1,1,length=1000), 1000, replace=FALSE), ncol=1) 
# The network will try to approximate the target P^2
target <- P^2                                   
#We create a feedforward network, with 2 neurons in the hidden layer. Tansig and Purelin activation functions.
net <- newff(n.inputs=1,n.hidden=2,n.outputs=1,learning.rate.global=1e-1, momentum.global=0.5 , error.criterium="MSE", hidden.layer="tansig", output.layer="purelin")
net <- train(net,P,target,n.epochs=100, g=adapt.NeuralNet,error.criterium="MSE", Stao=NA, report=TRUE, show.step=10 )
}
\keyword{neural}
