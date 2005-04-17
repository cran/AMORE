\encoding{latin1}
\name{ADAPTgdwm.Backwards.MLPneuron}
\alias{ADAPTgdwm.Backwards.MLPneuron}
\title{Backpropagates the single pattern error modifying accordingly the neuron's weights and bias in the ADAPTgdwm training method.}
\description{ This is the original R code version mainly written to be called by \link{ADAPTgdwm.Backwards.MLPnet.R}. Since this latter function has been rewritten (\link{ADAPTgdwm.Backwards.MLPnet}) to call the C code corresponding function, it has been kept just for educational purposes and it is not meant to be used otherwise.}
\usage{ ADAPTgdwm.Backwards.MLPneuron(net, ind.MLPneuron) }
\arguments{
\item{net}{Neural Network to train.}
\item{ind.MLPneuron}{Index of the neuron that needs to suffer the backpropagation effect.}
}
\value{This function returns a neuron object with weights and bias adjusted by the adaptative backpropagation with momentum method.}
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
\code{\link{ADAPTgdwm.Backwards.MLPnet}}, \code{\link{ADAPTgdwm.MLPnet}}, \code{\link{train}} 
}
\keyword{neural}

