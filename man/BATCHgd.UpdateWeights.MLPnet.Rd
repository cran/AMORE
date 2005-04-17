\encoding{latin1}
\name{BATCHgd.UpdateWeights.MLPnet}
\alias{BATCHgd.UpdateWeights.MLPnet}
\alias{BATCHgd.UpdateWeights.MLPnet.R}

\title{Update weights.}
\description{This function is used after the forward pass to change the values of the weights and the bias.}

\usage{
BATCHgd.UpdateWeights.MLPnet(net)
BATCHgd.UpdateWeights.MLPnet.R(net)
}

\arguments{
\item{net}{Neural Network that is being trained.}
}

\value{This function returns with the weights and bias values accordingly modified.}
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
\code{\link{BATCHgd.UpdateWeights.MLPneuron}}, \code{\link{train}} 
}


\keyword{neural}

