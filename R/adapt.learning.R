###############################################################################################################
#				ADAPT
##########################################################
forward.adapt.R.NeuralNet <- function(net,Pvector) {
net$layers$input.layer <- Pvector
for ( ind.layer in 2:length(net$layers) ) {
   for ( ind.neuron in 1:length(net$layers[[ind.layer]]) ) {
       this.neuron <- net$layers[[ind.layer]][[ind.neuron]]
       net$neurons[[this.neuron]] <- forward.adapt.R.neuron( net, this.neuron )
   }
}
return(net)
}
##########################################################
forward.adapt.C.NeuralNet <- function(net,Pvector) {
net$layers$input.layer <- Pvector
for ( ind.layer in 2:length(net$layers) ) {
   for ( ind.neuron in 1:length(net$layers[[ind.layer]]) ) {
       this.neuron <- net$layers[[ind.layer]][[ind.neuron]]
       net$neurons[[this.neuron]] <- .Call("ForwardAdaptNeuronC", net, as.integer(this.neuron), new.env(), PACKAGE="AMORE" )
   }
}
return(net)
}
##########################################################
forward.adapt.R.neuron <- function(net,ind.neuron) {

neuron <-    net$neurons[[ind.neuron]]

a <- 0
for (ind.weight in 1:length(neuron$weights)) {
   if (neuron$input.links[ind.weight] < 0 ) {
      x.input <- net$layers$input.layer[-neuron$input.links[ind.weight]]
   } else {
      x.input <- net$neurons[[neuron$input.links[ind.weight]]]$v0
   }
   a <- a + neuron$weights[ind.weight]*x.input
}
   a <- a + neuron$bias

neuron$v0 <- neuron$f0(a)
neuron$v1 <- switch(neuron$activation.function,
               "tansig"  = b.tansig/a.tansig*(a.tansig-neuron$v0)*(a.tansig+neuron$v0),
               "sigmoid" = a.sigmoid*neuron$v0*(1-neuron$v0),
               "purelin" = 1,
               "hardlim" = NA,
               "custom"  = neuron$f1(a)
             )
return(neuron)
}
##########################################################
backpropagate.adapt.R.NeuralNet <-  function(net,target) {
   net$target <- target
   for ( ind.layer in length(net$layers):2) {
      for (ind.neuron in 1:length(net$layers[[ind.layer]])) {
         this.neuron <- net$layers[[ind.layer]][[ind.neuron]]
         net$neurons[[this.neuron]] <- backpropagate.adapt.R.neuron(net, this.neuron)
      }
   }
   return(net)
}
##########################################################
backpropagate.adapt.C.NeuralNet <-  function(net,target) {
   net$target <- target
   for ( ind.layer in length(net$layers):2) {
      for (ind.neuron in 1:length(net$layers[[ind.layer]])) {
         this.neuron <- net$layers[[ind.layer]][[ind.neuron]]
         net$neurons[[this.neuron]] <- .Call("BackpropagateAdaptNeuronC", net, as.integer(this.neuron), new.env(), PACKAGE="AMORE" )
      }
   }
   return(net)
}

##########################################################
backpropagate.adapt.R.neuron <- function(net, ind.neuron) {
   neuron <- net$neurons[[ind.neuron]]
   if (neuron$type=="output") {
      aux.delta <- net$deltaE(list(neuron$v0, net$target[[neuron$output.aims[1]]], net$other.elements)) # una neurona de salida solo tiene un "aim"
   } else {
      # calculo de delta_j=f1*sum(w_{kj} * delta_k)
      aux.delta <- 0
      for ( ind.other.neuron in 1:length(neuron$output.links)) {
         that.neuron <- neuron$output.links[ind.other.neuron]
         that.aim    <- neuron$output.aims[ind.other.neuron]
         aux.delta   <- aux.delta + net$neurons[[that.neuron]]$weights[that.aim]*net$neurons[[that.neuron]]$delta
      }
   }
   neuron$delta                <- aux.delta*neuron$v1
   bias.change                 <- neuron$momentum * neuron$former.bias.change -  neuron$learning.rate * neuron$delta
   neuron$former.bias.change   <- bias.change
   neuron$bias                 <- neuron$bias + bias.change

   weight.change  <- rep(0,length(neuron$weights))
   for (ind.weight in 1:length(neuron$weights)) {
      if (neuron$input.links[ind.weight] < 0 ) {
        x.input <- net$layers$input.layer[-neuron$input.links[ind.weight]]
      } else {
        x.input <- net$neurons[[neuron$input.links[[ind.weight]]]]$v0
      }
      weight.change[ind.weight]  <- neuron$momentum * neuron$former.weight.change[ind.weight] - neuron$learning.rate * neuron$delta * x.input 
   }

   neuron$weights              <- neuron$weights + weight.change
   neuron$former.weight.change <- weight.change

   return(neuron)
}
##########################################################
adapt.R.NeuralNet <- function(net,Pvector,target) {
   nsalidas <- length(net$layer[[length(net$layer)]])
   y <- rep(0, nsalidas)
   net <- forward.adapt.R.NeuralNet(net,Pvector)
   net <- backpropagate.adapt.R.NeuralNet(net,target)   
   return(net)
}

##########################################################
adapt.C1.NeuralNet <- function(net,Pvector,target) {
   nsalidas <- length(net$layer[[length(net$layer)]])
   y <- rep(0, nsalidas)
   net <- forward.adapt.C.NeuralNet(net,Pvector)
   net <- backpropagate.adapt.R.NeuralNet(net,target)   
   return(net)
}

##################################################
adapt.C2.NeuralNet <- function(net,Pvector,target) {
   nsalidas <- length(net$layer[[length(net$layer)]])
   y <- rep(0, nsalidas)
   net <- .Call("ForwardAdaptNeuralNetC", net, Pvector , new.env(), PACKAGE="AMORE" )
   net <- backpropagate.adapt.R.NeuralNet(net,target)   
   return(net)
}
##################################################
adapt.C3.NeuralNet <- function(net,Pvector,target) {
   nsalidas <- length(net$layer[[length(net$layer)]])
   y <- rep(0, nsalidas)
   net <- .Call("ForwardAdaptNeuralNetC", net, Pvector , new.env(), PACKAGE="AMORE" )
   net <- backpropagate.adapt.C.NeuralNet(net, target)
   return(net)
}
##################################################
adapt.NeuralNet <- function(net,Pvector,target) {
   nsalidas <- length(net$layer[[length(net$layer)]])
   y <- rep(0, nsalidas)
   net <- .Call("ForwardAdaptNeuralNetC", net, Pvector , new.env(), PACKAGE="AMORE" )
   net <- .Call("BackpropagateAdaptNeuralNetC", net, target, new.env(), PACKAGE="AMORE")   
   return(net)
}

##################################################




