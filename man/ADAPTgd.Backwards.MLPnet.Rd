\encoding{latin1}
\name{ADAPTgd.Backwards.MLPnet}
\alias{ADAPTgd.Backwards.MLPnet}
\alias{ADAPTgd.Backwards.MLPnet.R}


\title{Backpropagates the single pattern error modifying accordingly the Neural Network's weights and biases in the ADAPTgd training method.}
\description{ Mainly written to be called by the ADAPTgd.MLPnet function.}

\usage{
ADAPTgd.Backwards.MLPnet(net,target)
ADAPTgd.Backwards.MLPnet.R(net,target)
}

\arguments{
\item{net}{Neural Network to train.}
\item{target}{Training output vector.}
}

\value{This function returns a Neural Network object with weights and biases adjusted by the adaptative backpropagation (without momentum) method.}
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
\code{\link{ADAPTgd.MLPnet}}, \code{\link{train}}
}


\keyword{neural}

