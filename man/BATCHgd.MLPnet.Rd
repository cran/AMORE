\encoding{latin1}
\name{BATCHgd.MLPnet}
\alias{BATCHgd.MLPnet}
\alias{BATCHgd.MLPnet.R}
\title{Batch gradient descent (without momentum) training}
\description{Modifies the neural network weights and biases according to the training set. Mostly written to be called by the \link{train}.}
\usage{
BATCHgd.MLPnet(net,P,T)
BATCHgd.MLPnet.R(net,P,T)
}

\arguments{
\item{net}{Neural Network to train.}
\item{P}{Input values of the training pattern.}
\item{T}{Target values to reach at the network outputs.}
}

\value{These functions return a neural network object modified according to the chosen data.}
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

