\encoding{latin1}
\name{BATCHgd.Forward.MLPnet}
\alias{BATCHgd.Forward.MLPnet}
\alias{BATCHgd.Forward.MLPnet.R}
\title{Performs the forward pass in the BATCHgd training method.}
\description{Written to be called by the \code{\link{BATCHgd.MLPnet}} function. }

\usage{
BATCHgd.Forward.MLPnet(net,P, T)
BATCHgd.Forward.MLPnet.R(net,P, T)
 }

\arguments{
\item{net}{Neural Network to adapt.}
\item{P}{Input training set to propagate.}
\item{T}{Target data set to approximate.}
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
\code{\link{BATCHgd.Forward.MLPneuron}}, \code{\link{BATCHgd.MLPnet}}
}


\keyword{neural}

