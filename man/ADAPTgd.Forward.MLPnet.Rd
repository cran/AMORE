\encoding{latin1}
\name{ADAPTgd.Forward.MLPnet}
\alias{ADAPTgd.Forward.MLPnet}
\alias{ADAPTgd.Forward.MLPnet.R}
\title{Performs the forward pass in the ADAPTgd training method.}
\description{Written to be called by the \code{\link{train}} function. This function is similar to \code{\link{sim.Forward.MLPnet.R}} but the derivative of the activation function of the Neural Network neurons are also calculated.  Both R and C versions are provided.}

\usage{
ADAPTgd.Forward.MLPnet(net,Pvector)
ADAPTgd.Forward.MLPnet.R(net,Pvector)
 }

\arguments{
\item{net}{Neural Network to adapt.}
\item{Pvector}{Input vector to propagate.}
}

\value{This function returns the forward passed Neural Network.}
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
\code{\link{ADAPTgd.Forward.MLPneuron}}
}

\keyword{neural}

