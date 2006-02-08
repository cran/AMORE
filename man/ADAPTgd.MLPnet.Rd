\encoding{latin1}
\name{ADAPTgd.MLPnet}
\alias{ADAPTgd.MLPnet}
\title{Adaptative gradient descent training}
\description{Adaptative gradient descent training method.}
\usage{
ADAPTgd.MLPnet(net,P, T,n.epochs)
}

\arguments{
\item{net}{Neural Network to train.}
\item{P}{Input data set.}
\item{T}{Target output data set.}
\item{n.epochs}{Number of epochs to train}
}

\value{This function returns a neural network object modified according to the input and target data set.}
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
\code{\link{newff},\link{train},\link{ADAPTgdwm.MLPnet}} 
}



\keyword{neural}
