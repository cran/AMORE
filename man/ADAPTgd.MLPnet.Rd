\encoding{latin1}
\name{ADAPTgd.MLPnet}
\alias{ADAPTgd.MLPnet}
\alias{ADAPTgd.MLPnet.R}
\title{Adaptative on-line gradient descent single pattern training}
\description{Modifies the neural network weights and biases according to one pattern chosen from the training set. Mostly written to be called by the \link{train} function so as to be applied to the whole training set. The user may choose amongst two versions of the same function: The slower ADAPTgd.MLPnet.R was the original R code function and ADAPTgd.MLPnet is the C conversed function and the one that provides higher speeds of execution.}
\usage{
ADAPTgd.MLPnet(net,Pvector,target)
ADAPTgd.MLPnet.R(net,Pvector,target)
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
\code{\link{train}} 
}


\keyword{neural}

