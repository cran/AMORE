\encoding{latin1}
\name{ADAPTgdwm.Backwards.MLPnet}
\alias{ADAPTgdwm.Backwards.MLPnet}
\alias{ADAPTgdwm.Backwards.MLPnet.R}


\title{Backpropagates the single pattern error modifying accordingly the Neural Network's weights and biases in the ADAPTgdwm training method.}
\description{ Mainly written to be called by the ADAPTgdwm.MLPnet function.}

\usage{
ADAPTgdwm.Backwards.MLPnet(net,target)
ADAPTgdwm.Backwards.MLPnet.R(net,target)
}

\arguments{
\item{net}{Neural Network to train.}
\item{target}{Training output vector.}
}

\value{This function returns a Neural Network object with weights and biases adjusted by the adaptative backpropagation with momentum method.}
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
\code{\link{ADAPTgdwm.MLPnet}}, \code{\link{train}}
}


\keyword{neural}

