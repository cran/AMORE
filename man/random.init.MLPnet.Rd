\encoding{latin1}
\name{random.init.MLPnet}
\alias{random.init.MLPnet}
\title{Initialize the network with random weigths and biases.}
\description{Provides random values to the network weights and biases so as to start with. Basically it applies the random.init.MLPneuron function to every neuron in the network.}
\usage{random.init.MLPnet(net) }
\arguments{ \item{net}{The neural network object} }
\value{\emph{random.init.MLPnet} returns the input network with weights and biases changed randomly.}
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
\code{\link{random.init.MLPneuron}}, \code{\link{init.MLPneuron}}, \code{\link{newff}}
}
\keyword{neural}