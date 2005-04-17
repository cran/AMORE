\encoding{latin1}
\name{ADAPTgd.Forward.MLPneuron}
\alias{ADAPTgd.Forward.MLPneuron}

\title{Perform the neuron forward pass in the ADAPTgd training method.}
\description{This function is similar to \code{\link{sim.Forward.MLPneuron}} but the derivative of the activation function of the neuron is also calculated. }

\usage{
ADAPTgd.Forward.MLPneuron(net,ind.MLPneuron)
}

\arguments{
\item{net}{Neural Network to adapt.}
\item{ind.MLPneuron}{Index of the neuron to forward pass.}
}

\value{This function returns the forward passed neuron.}
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
\code{\link{ADAPTgd.Forward.MLPnet}}}


\keyword{neural}

