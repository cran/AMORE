\encoding{latin1}
\name{error.TAO}
\alias{error.MSE}
\alias{error.LMLS}
\alias{error.TAO}
\alias{deltaE.MSE}
\alias{deltaE.LMLS}
\alias{deltaE.TAO}

\title{Neural network training error criteria.}

\description{The error functions calculate the goodness of fit of a neural network according to certain criterium:
\item MSE:  Mean Squared Error. Least Mean Squares minimization.
\item LMLS: Least Mean Log Squares minimization. 
\item TAO:  TAO error minimization.
The deltaE functions calculate the corresponding influence functions.
}

\usage{
error.MSE(arguments)
error.LMLS(arguments)
error.TAO(arguments)
deltaE.MSE(arguments)
deltaE.LMLS(arguments)
deltaE.TAO(arguments)
}
\arguments{
\item{arguments}{List of arguments to pass to the functions.
   \item The first element is the prediction of the neural network.
   \item The second element is the target value.
   \item A third element is needed for the TAO method containing the value of the S parameter.}
}

\value{This functions return the error and influence function criteria.}
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
Pernia Espinoza, A.V. TAO-robust backpropagation learning algorithm. Neural Networks. In press. \cr \cr
Simon Haykin. Neural Networks. A comprehensive foundation. 2nd Edition. \cr \cr
}

\seealso{
\code{\link{train}}, \code{\link{train.compare}}
}

\keyword{neural}

