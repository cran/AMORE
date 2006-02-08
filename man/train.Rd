\encoding{latin1}
\name{train}
\alias{train}

\title{Neural network training function.}

\description{For a given data set (training set), this function modifies the neural network weights and biases to approximate the relationships amongst variables present in the training set. These may serve to satisfy several needs, i.e. fitting non-linear functions.}

\usage{
train(net, P, T, Pval=NULL, Tval=NULL, error.criterium="LMS", report=TRUE, n.shows, show.step, Stao=NA,prob=NULL)
}

\arguments{
\item{net}{Neural Network to train.}
\item{P}{Training set input values.}
\item{T}{Training set output values}
\item{Pval}{Validation set input values for optional early stopping.}
\item{Tval}{Validation set output values for optional early stopping.}
\item{error.criterium}{Criterium used to measure the goodness of fit:"LMS", "LMLS", "TAO".}
\item{Stao}{Initial value of the S parameter used by the TAO algorithm.}
\item{report}{Logical value indicating whether the training function should keep quiet or should provide graphical/written information during the training process instead.}
\item{n.shows}{Number of times to report (if report is TRUE). The total number of training epochs is n.shows times show.step.}
\item{show.step}{Number of epochs to train non-stop until the training function is allow to report.}
\item{prob}{Vector with the probabilities of each sample so as to apply resampling training.}
}

\value{This function returns a list with two elements: the trained Neural Network object with weights and biases adjusted by the adaptative backpropagation with momentum method and a matrix with the errors obtained during the training. If the validation set is provided, the early stopping technique is applied.}
\author{
Manuel Castej� Limas.             \email{manuel.castejon@unileon.es}\cr
Joaquin Ordieres Mer�            \email{joaquin.ordieres@dim.unirioja.es}\cr
Ana Gonz�ez Marcos.               \email{ana.gonzalez@unileon.es} \cr
Alpha V. Pern� Espinoza.          \email{alpha.pernia@alum.unirioja.es}\cr
Eliseo P. Vergara Gonzalez.       \email{eliseo.vergara@dim.unirioja.es}\cr
Francisco Javier Martinez de Pis�. \email{francisco.martinez@dim.unirioja.es}\cr
Fernando Alba El�s.                \email{fernando.alba@unavarra.es}\cr
}

\references{
Pernia Espinoza, A.V. TAO-ROBUST BACKPROPAGATION LEARNING ALGORITHM  \cr \cr
Simon Haykin. Neural Networks. A comprehensive foundation. 2nd Edition. \cr \cr
}

\seealso{
\code{\link{newff}} } 
}

\keyword{neural}

