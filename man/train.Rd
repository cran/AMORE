\encoding{latin1}
\name{train}
\alias{train}

\title{Neural network training function.}

\description{For a given data set (training set), this function modifies the neural network weights and biases to approximate the relationships amongst variables present in the training set. These may serve to satisfy several needs, i.e. fitting non-linear functions.}

\usage{
train(net, P, T, n.epochs, error.criterium="LMS", report=TRUE, show.step, Stao=NA)
}

\arguments{
\item{net}{Neural Network to train.}
\item{P}{Training set input values.}
\item{T}{Training set output values}
\item{n.epochs}{Number of epochs to train.}
\item{error.criterium}{Criterium used to measure the goodness of fit: "LMS", "LMLS", "TAO".}
\item{Stao}{Initial value of the S parameter used by the TAO algorithm.}
\item{report}{Logical value indicating whether the training function should keep quiet or should provide graphical/written information during the training process instead.}
\item{show.step}{If report is TRUE then a report is provided every show.step epochs.}
}

\value{This function returns a trained Neural Network object with weights and biases adjusted by the adaptative backpropagation with momentum method. The whole training set is considered.}
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
Pernia Espinoza, A.V. TAO-ROBUST BACKPROPAGATION LEARNING ALGORITHM  \cr \cr
Simon Haykin. Neural Networks. A comprehensive foundation. 2nd Edition. \cr \cr
}

\seealso{
\code{\link{newff}} } 
}

\keyword{neural}

