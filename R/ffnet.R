
#################################
# Creates a new NeuralNet object
#############################
newff <- function (n.inputs, n.hidden, n.outputs, learning.rate.global, momentum.global=NA, error.criterium="MSE", Stao=NA, hidden.layer=c("purelin","tansig","sigmoid","hardlim"), output.layer=c("purelin","tansig","sigmoid","hardlim")) {
   possible.activation.functions <- c("purelin","tansig","sigmoid","hardlim")
   net         <- list( layers=list(), neurons=list() , target=0, deltaE=NA, other.elements=list() )

   if ( is.na(hidden.activation.function.choice <- pmatch(hidden.layer,possible.activation.functions)) ) {
       stop("You should use a correct activation function for the hidden layer.")
   } else {
      for (ind.neuron in 1:n.hidden) {
         net$neurons[[ind.neuron]] <- init.neuron(id=ind.neuron,type="hidden",activation.function=possible.activation.functions[hidden.activation.function.choice],output.links=(n.hidden+1):(n.hidden+n.outputs), output.aims=rep(ind.neuron, n.outputs), input.links=-c(1:n.inputs),weights=rep(0,n.inputs), bias=0, former.weight.change=rep(0,n.inputs), former.bias.change=0,learning.rate=learning.rate.global, sum.delta.x=rep(0,n.inputs), sum.delta.bias=0, momentum=momentum.global, delta=0 )
      }
   }

   if ( is.na(output.activation.function.choice <- pmatch(output.layer,possible.activation.functions)) ) {
       stop("You should use a correct activation function for the output layer.")
   } else {
      for (ind.neuron in (n.hidden+1):(n.hidden+n.outputs)) {
         net$neurons[[ind.neuron]] <- init.neuron(id=ind.neuron,type="output",activation.function=possible.activation.functions[output.activation.function.choice],output.links=NA, output.aims=ind.neuron-n.hidden, input.links=1:n.hidden,weights=rep(0,n.hidden), bias=0, former.weight.change=rep(0,n.hidden), former.bias.change=0, learning.rate=learning.rate.global, sum.delta.x=rep(0,n.hidden), sum.delta.bias=0, momentum=momentum.global, delta=0 )
      } 
   }

   net$layers$input.layer  <- rep(0,n.inputs)
   net$layers$hidden.layer <- 1:n.hidden 
   net$layers$output.layer <- (n.hidden+1):(n.hidden+n.outputs)

   if (error.criterium == "MSE" ) {
      net$deltaE <- deltaE.MSE
   } else if (error.criterium == "LMLS") {
       net$deltaE <- deltaE.LMLS
   } else if (error.criterium == "TAO") {   
       net$deltaE <- deltaE.TAO 
       if (missing(Stao)){ 
          stop("You should enter the Stao value")
       } else {
          net$other.elements$Stao <-Stao
       }
   } else {
      stop("You should enter either: \"MSE\", \"LMSL\" or \"TAO\". ")
   }
   class(net)              <- "NeuralNet"
   net <- random.init.NeuralNet(net)
return(net)
}
#################################
# Creates individual neurons
#########################
init.neuron   <- function(id,type,activation.function,output.links, output.aims, input.links, weights, bias, former.weight.change, former.bias.change, learning.rate, sum.delta.x, sum.delta.bias, momentum, delta) {
aux <- select.activation.function(activation.function)
neuron                      <- list()
neuron$id                   <- as.integer(id)
neuron$type                 <- as.character(type)
neuron$activation.function  <- activation.function
neuron$output.links         <- as.integer(output.links)
neuron$output.aims          <- as.integer(output.aims)
neuron$input.links          <- as.integer(input.links)
neuron$weights              <- as.double(weights)
neuron$former.weight.change <- as.double(former.weight.change)
neuron$bias                 <- as.double(bias)
neuron$former.bias.change   <- as.double(former.bias.change)
neuron$v0                   <- as.double(0)
neuron$v1                   <- as.double(0)
neuron$learning.rate        <- as.double(learning.rate)
neuron$sum.delta.x          <- as.double(sum.delta.x)
neuron$sum.delta.bias       <- as.double(sum.delta.bias)
neuron$momentum             <- as.double(momentum)
neuron$f0                   <- aux$f0
neuron$f1                   <- aux$f1
neuron$delta                <- as.double(delta)
class(neuron) <- "neuron"
return(neuron)
}
#########################################
# Initialize the neuron bias and weights with random values according to the book:
# Neural Networks. A comprehensive foundation. 2nd Edition.
# Author: Simon Haykin.
# pages = 182, 183, 184.
#################################
random.init.neuron <- function(net.number.weights, neuron) {
   extreme        <- sqrt(3/net.number.weights)
   n.weights      <- length(neuron$weights)
   neuron$weights <- runif(n.weights,min=-extreme,max=extreme)
   neuron$bias    <- runif(1,min=-extreme,max=extreme)
   return(neuron)
}
#################################################
# Runs random.init.neuron upon each neuron.
###########################################
random.init.NeuralNet <- function(net) {
   net.number.weights <- length(net$neurons)          #number of bias terms
   for (ind.neuron in 1:length(net$neurons)) {
          net.number.weights <- net.number.weights + length(net$neurons[[ind.neuron]]$weights)
       }

   for ( i in 1:length(net$neurons)) { 
      net$neurons[[i]] <- random.init.neuron(net.number.weights,net$neurons[[i]] )
   }
return(net)
}

#########################################
# A simple function to bestow the neuron with the appropriate 
select.activation.function <- function(activation.function) {
   f0 <- NA
   f1 <- NA
   if (activation.function == "tansig" ) {
     f0 <- function (v) {
                          a.tansig   <- 1/tanh(2/3)
                          b.tansig   <- 2/3
                          return ( a.tansig * tanh( v * b.tansig ) )
                        }
     f1 <- function (v) {         # realmente usaremos f1= b.tansig/a.tansig*(a.tansig-f0)*(a.tansig+f0)
                          a.tansig   <- 1/tanh(2/3)
                          b.tansig   <- 2/3
                          return( a.tansig * b.tansig * (1-tanh( v * b.tansig )^2)  )
                        } 
    } else if (activation.function == "sigmoid" ) {
     f0 <- function (v) {
                          a.sigmoid  <- 1
                          return( 1/(1+exp(- a.sigmoid * v)) )
                        }
     f1 <- function (v) {           # realmente usaremos f1=a.sigmoid*f0*(1-f0)
                          a.sigmoid  <- 1
                          return ( a.sigmoid * exp(- a.sigmoid * v) / (1+exp(- a.sigmoid * v))^2 )
                        } 
   } else if (activation.function == "hardlim" ) {
     f0 <- function (v) {
                          if (v>=0) { return(1) } else { return(0) }
                        }
     f1 <- function (v) {
                          return ( NA )
                        }
   } else if (activation.function == "purelin" ) {
     f0 <- function (v) {
                          return( v )  
                        }
     f1 <- function (v) {
                          return( 1 ) 
                        }
   }
   return(list(f0=f0,f1=f1))
}

##############################################################
# Manually set the learning rate and momentum for each neuron
##############################################################

set.learning.rate.and.momentum <- function(net, learning.rate, momentum) {
   for (i in 1:length(net$neurons)) {
      net$neurons[[i]]$learning.rate <- learning.rate
      net$neurons[[i]]$momentum <- momentum
   }
   return(net)
}



