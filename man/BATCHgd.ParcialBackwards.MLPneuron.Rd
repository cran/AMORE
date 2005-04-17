\encoding{latin1}
\name{BATCHgd.ParcialBackwards.MLPneuron}
\alias{BATCHgd.ParcialBackwards.MLPneuron}

\title{Parcial Backwards.}
\description{This function is used during the forward pass to take into account the residual of the current sample and to accumulate the corresponding weight and bias changes.}

\usage{BATCHgd.ParcialBackwards.MLPneuron(net, ind.MLPneuron)}

\arguments{
\item{net}{Neural Network to train.}
\item{ind.MLPneuron}{Index of the neuron that needs to suffer the backpropagation effect.}
}

\value{This function returns a neuron object with the \emph{delta}, \emph{sum.delta.x} and \emph{sum.delta.bias} accordingly modified.}
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
\code{\link{ADAPTgd.Backwards.MLPnet}}, \code{\link{ADAPTgd.MLPnet}}, \code{\link{train}} 
}


\keyword{neural}

