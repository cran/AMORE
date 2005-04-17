################################################################################################################
#	BATCHgd ( BATCH gradient descent without momentum )
##############################################################
BATCHgd.MLPnet <- function(net, P, T) { # Each pattern is a row of P, 
   net <- BATCHgd.Forward.MLPnet(net, P, T)
   net <- .Call("BATCHgd_UpdateWeights_MLPnet", net, PACKAGE="AMORE")   
   return(net)
}
##############################################################
BATCHgd.MLPnet.R <- function(net, P, T) { # Each pattern is a row of P, 
   net <- BATCHgd.Forward.MLPnet.R(net, P, T)
   net <- BATCHgd.UpdateWeights.MLPnet.R(net)   
   return(net)
}
##############################################################
BATCHgd.Forward.MLPnet.R <- function(net, P, T) {
   for (ind.MLPneuron in 1:length(net$neurons)) {
      net$neurons[[ind.MLPneuron]]$method.dep.variables$sum.delta.bias <- as.double(0)
      net$neurons[[ind.MLPneuron]]$method.dep.variables$sum.delta.x    <- as.double(rep(0,length(net$neurons[[ind.MLPneuron]]$method.dep.variables$sum.delta.x)))
   }
   for (ind.sample in 1:nrow(P)) {
      net$input <- P[ind.sample,]
      net$target             <- T[ind.sample,]
      # In the first pass we propagate the inputs.
      for ( ind.layer in 2:length(net$layers) ) {
         for ( ind.MLPneuron in 1:length(net$layers[[ind.layer]]) ) {
            this.MLPneuron <- net$layers[[ind.layer]][[ind.MLPneuron]]
            net$neurons[[this.MLPneuron]] <- BATCHgd.Forward.MLPneuron( net, this.MLPneuron )
         }
      }
      # In the second pass we sum the corresponding terms of sum.delta.x
      for ( ind.layer in length(net$layers):2) {
         for ( ind.MLPneuron in length(net$layers[[ind.layer]]):1 ) {
            this.MLPneuron <- net$layers[[ind.layer]][[ind.MLPneuron]]
            net$neurons[[this.MLPneuron]] <- BATCHgd.ParcialBackwards.MLPneuron(net, this.MLPneuron)
         }
      }
   }
   return(net)
}
##############################################################
BATCHgd.Forward.MLPnet <- function(net,P,T) {
   for (ind.MLPneuron in 1:length(net$neurons)) {
      net$neurons[[ind.MLPneuron]]$method.dep.variables$sum.delta.bias <- as.double(0)
      net$neurons[[ind.MLPneuron]]$method.dep.variables$sum.delta.x    <- as.double(rep(0,length(net$neurons[[ind.MLPneuron]]$method.dep.variables$sum.delta.x)))
   }
   for (ind.sample in 1:nrow(P)) {
      net$input  <- P[ind.sample,]
      net$target <- T[ind.sample,]
  # In the first pass we propagate the inputs.
      net <- .Call("BATCHgd_ForwardPass_MLPnet",      net, new.env(), PACKAGE="AMORE" )
  # In the second pass we sum the corresponding terms of sum.delta.x
#save(file=paste("Red-",ind.sample,".RData",sep=""), net)
#cat("Guardada",paste("Red-",ind.sample,".RData",sep=""),"\n" )
      net <- .Call("BATCHgd_ParcialBackwards_MLPnet", net, new.env(), PACKAGE="AMORE" )
   }
   return(net)
}
##########################################################
BATCHgd.Forward.MLPneuron <- function(net,ind.MLPneuron) {
   neuron     <- net$neurons[[ind.MLPneuron]]
   a <- 0
   for (ind.weight in 1:length(neuron$weights)) {
      if (neuron$input.links[ind.weight] < 0 ) {
         x.input <- net$input[-neuron$input.links[ind.weight]]
      } else {
         x.input <- net$neurons[[neuron$input.links[ind.weight]]]$v0
      }
      a <- a + neuron$weights[ind.weight]*x.input
   }
   a <- a + neuron$bias
   neuron$v0 <- neuron$f0(a)

   a.tansig   <- 1.715904708575539
   b.tansig   <- 0.6666666666666667
   b.split.a  <- 0.3885219635652736
   a.sigmoid  <- 1.0
   neuron$v1 <- switch(neuron$activation.function,
               "tansig"  = b.split.a*(a.tansig-neuron$v0)*(a.tansig+neuron$v0),
               "sigmoid" = a.sigmoid*neuron$v0*(1-neuron$v0),
               "purelin" = 1,
               "hardlim" = NA,
               "custom"  = neuron$f1(a)
             )
return(neuron)
}
##########################################################
BATCHgd.ParcialBackwards.MLPneuron <- function(net, ind.MLPneuron) {
   neuron <- net$neurons[[ind.MLPneuron]]
   if (neuron$type=="output") {
      aux.delta <- net$deltaE( list(prediction=neuron$v0, target=net$target[[neuron$output.aims[1]]], net) )  # una neurona de salida solo tiene un "aim"
#meter 3 parámetro
   } else {
      # calculo de delta_j=f1*sum(w_{kj} * delta_k)
      aux.delta <- 0
      for ( ind.other.MLPneuron in 1:length(neuron$output.links)) {
        that.MLPneuron <- neuron$output.links[ind.other.MLPneuron]
        that.aim       <- neuron$output.aims[ind.other.MLPneuron]
        aux.delta      <- aux.delta + net$neurons[[that.MLPneuron]]$weights[that.aim]*net$neurons[[that.MLPneuron]]$method.dep.variables$delta
      }
   }
   neuron$method.dep.variables$delta   <- neuron$v1 * aux.delta
      
   #Corrección de los pesos

   for(ind.weight in 1:length(neuron$weights)) {
      if (neuron$input.links[ind.weight] < 0 ) {
         x.input <- net$input[-neuron$input.links[ind.weight]]
      } else {
         x.input <- net$neurons[[neuron$input.links[[ind.weight]]]]$v0
      }
      neuron$method.dep.variables$sum.delta.x[ind.weight] <- neuron$method.dep.variables$sum.delta.x[ind.weight] + neuron$method.dep.variables$delta * x.input    
   }

  neuron$method.dep.variables$sum.delta.bias <- neuron$method.dep.variables$sum.delta.bias + neuron$method.dep.variables$delta
  return(neuron)
}
##########################################################
##########################################################
BATCHgd.UpdateWeights.MLPnet.R <-  function(net) {
   for ( ind.MLPneuron in 1:length(net$neurons)) {
       net$neurons[[ind.MLPneuron]] <- BATCHgd.UpdateWeights.MLPneuron(net, ind.MLPneuron)
   }
   return(net)
}
##########################################################
BATCHgd.UpdateWeights.MLPnet <-  function(net) {
   for ( ind.MLPneuron in 1:length(net$neurons)) {
       net$neurons[[ind.MLPneuron]] <-  .Call("BATCHgd_UpdateWeights_MLPneuron", net, as.integer(ind.MLPneuron), PACKAGE="AMORE" )
   }
   return(net)
}
##########################################################
BATCHgd.UpdateWeights.MLPneuron <- function(net, ind.MLPneuron) {
   neuron <- net$neurons[[ind.MLPneuron]]
   for (ind.weight in 1:length(neuron$weights)) {
      weight.change <-  - neuron$method.dep.variables$learning.rate * neuron$method.dep.variables$sum.delta.x[ind.weight]
      neuron$weights[ind.weight] <- neuron$weights[ind.weight] + weight.change
   }
   bias.change <-  - neuron$method.dep.variables$learning.rate * neuron$method.dep.variables$sum.delta.bias
   neuron$bias <- neuron$bias + bias.change
   return(neuron)
}
##########################################################

