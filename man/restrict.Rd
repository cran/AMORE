\encoding{latin1}
\name{restrict}
\alias{restrict}

\title{Restrict the patterns to the [-1,1] interval.}

\description{This function centers and scales the data set so as to be confined in the [-1,1] interval. Then no x.max nor x.min should be indicated. In case new data is considered, the user may apply the same transformation according to the limits of the previous data set, thus obtaining, for example, the image of a new data set in the interval used for training.}

\usage{
restrict(x,x.max,x.min) 
}

\arguments{
\item{x}{Data set to be centered and scaled.}
\item{x.max}{Upper bound of a previous data set.}
\item{x.min}{Lower bound of a previous data set.}
}

\value{This function returns a list. The first element contains the transformed x values. The second element contains x.max and the third x.min.}
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
\code{\link{train}}, \code{\link{train.compare}}
}

\keyword{neural}

