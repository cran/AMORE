\encoding{latin1}
\name{ADAPTgdwm.Forward.MLPnet}
\alias{ADAPTgdwm.Forward.MLPnet}
\alias{ADAPTgdwm.Forward.MLPnet.R}
\title{Performs the forward pass in the ADAPTgdwm training method.}
\description{Written to be called by the \code{\link{ADAPTgdwm.MLPnet}} function. }

\usage{
ADAPTgdwm.Forward.MLPnet(net,Pvector)
ADAPTgdwm.Forward.MLPnet.R(net,Pvector)
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
\code{\link{ADAPTgdwm.Forward.MLPneuron}}, \code{\link{ADAPTgdwm.MLPnet}}
}


\keyword{neural}

