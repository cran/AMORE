##########################################################
#	Adaptative Gradient Descent with Momentum
##########################################################

##########################################################
#   MLPnet level   Forward and Backward Passes
##########################################################

#########################################################################################################
ADAPTgdwm.Forward.MLPnet.R <- function(net,Pvector) {
##########################################################
########################################################## R version of: ADAPTgdwm_Forward_MLPnet 
##########################################################
net$input <- Pvector
for ( ind.layer in 2:length(net$layers) ) {
   for ( ind.MLPneuron in 1:length(net$layers[[ind.layer]]) ) {
       this.MLPneuron <- net$layers[[ind.layer]][[ind.MLPneuron]]
       net$neurons[[this.MLPneuron]] <- ADAPTgdwm.Forward.MLPneuron( net, this.MLPneuron )
   }
}
return(net)
}
#########################################################################################################
ADAPTgdwm.Forward.MLPnet <- function(net,Pvector) { 
##########################################################
########################################################## R version of:     ADAPTgdwm_Forward_MLPnet
########################################################## calling C funtion ADAPTgdwm_Forward_MLPneuron
##########################################################
net$input <- Pvector
for ( ind.layer in 2:length(net$layers) ) {
   for ( ind.MLPneuron in 1:length(net$layers[[ind.layer]]) ) {
       this.MLPneuron <- net$layers[[ind.layer]][[ind.MLPneuron]]
       net$neurons[[this.MLPneuron]] <- .Call("ADAPTgdwm_Forward_MLPneuron", net, as.integer(this.MLPneuron), new.env(), PACKAGE="AMORE" )
   }
}
return(net)
}
#########################################################################################################
ADAPTgdwm.Backwards.MLPnet.R <-  function(net,target) {
##########################################################
########################################################## R version of:     ADAPTgdwm_Backwards_MLPnet
##########################################################
   net$target <- target
   for ( ind.layer in length(net$layers):2) {
      for (ind.MLPneuron in 1:length(net$layers[[ind.layer]])) {
         this.MLPneuron <- net$layers[[ind.layer]][[ind.MLPneuron]]
         net$neurons[[this.MLPneuron]] <- ADAPTgdwm.Backwards.MLPneuron(net, this.MLPneuron)
      }
   }
   return(net)
}
#########################################################################################################
ADAPTgdwm.Backwards.MLPnet <-  function(net,target) {
##########################################################
########################################################## R version of:     ADAPTgdwm_Backwards_MLPnet
########################################################## calling C funtion ADAPTgdwm_Backwards_MLPneuron
##########################################################
   net$target <- target
   for ( ind.layer in length(net$layers):2) {
      for (ind.MLPneuron in 1:length(net$layers[[ind.layer]])) {
         this.MLPneuron <- net$layers[[ind.layer]][[ind.MLPneuron]]
         net$neurons[[this.MLPneuron]] <- .Call("ADAPTgdwm_Backwards_MLPneuron", net, as.integer(this.MLPneuron), new.env(), PACKAGE="AMORE" )
      }
   }
   return(net)
}

##########################################################
#   MLPneuron level   Forward and Backward Passes
##########################################################

#########################################################################################################
ADAPTgdwm.Forward.MLPneuron <- function(net,ind.MLPneuron) {
##########################################################
##########################################################  R version of: ADAPTgdwm_Forward_MLPneuron
##########################################################
a.tansig   <- 1.715904708575539
b.tansig   <- 0.6666666666666667
b.split.a  <- 0.3885219635652736
a.sigmoid  <- 1.0
neuron    <- net$neurons[[ind.MLPneuron]]
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
neuron$v1 <- switch(neuron$activation.function,
               "tansig"  = b.split.a * (a.tansig-neuron$v0)*(a.tansig+neuron$v0),
               "sigmoid" = a.sigmoid*neuron$v0*(1-neuron$v0),
               "purelin" = 1,
               "hardlim" = NA,
               "custom"  = neuron$f1(a)
             )
return(neuron)
}
#########################################################################################################
ADAPTgdwm.Backwards.MLPneuron <- function(net, ind.MLPneuron) {
##########################################################
##########################################################  R version of: ADAPTgdwm_Backwards_MLPneuron
########################################################## 
   neuron <- net$neurons[[ind.MLPneuron]]
   if (neuron$type=="output") {
      aux.delta <- net$deltaE(list(neuron$v0, net$target[[neuron$output.aims[1]]], net)) # una neurona de salida solo tiene un "aim"
   } else {
      # calculo de delta_j=f1*sum(w_{kj} * delta_k)
      aux.delta <- 0
      for ( ind.other.MLPneuron in 1:length(neuron$output.links)) {
         that.MLPneuron <- neuron$output.links[ind.other.MLPneuron]
         that.aim    <- neuron$output.aims[ind.other.MLPneuron]
         aux.delta   <- aux.delta + net$neurons[[that.MLPneuron]]$weights[that.aim]*net$neurons[[that.MLPneuron]]$method.dep.variables$delta
      }
   }
   neuron$method.dep.variables$delta <- aux.delta*neuron$v1
   bias.change                                      <- neuron$method.dep.variables$momentum * neuron$method.dep.variables$former.bias.change -  neuron$method.dep.variables$learning.rate * neuron$method.dep.variables$delta
   neuron$method.dep.variables$former.bias.change   <- bias.change
   neuron$bias                                      <- neuron$bias + bias.change

   weight.change  <- rep(0,length(neuron$weights))
   for (ind.weight in 1:length(neuron$weights)) {
      if (neuron$input.links[ind.weight] < 0 ) {
        x.input <- net$input[-neuron$input.links[ind.weight]]
      } else {
        x.input <- net$neurons[[neuron$input.links[[ind.weight]]]]$v0
      }
      weight.change[ind.weight]  <- neuron$method.dep.variables$momentum * neuron$method.dep.variables$former.weight.change[ind.weight] - neuron$method.dep.variables$learning.rate * neuron$method.dep.variables$delta * x.input 
   }

   neuron$weights              <- neuron$weights + weight.change
   neuron$method.dep.variables$former.weight.change <- weight.change

   return(neuron)
}
#########################################################################################################
ADAPTgdwm.MLPnet <- function(net,Pvector,target) {
##########################################################
########################################################## calling C funtion ADAPTgdwm_Forward_MLPnet
########################################################## calling C funtion ADAPTgdwm_Backwards_MLPnet
########################################################## 
   nsalidas <- length(net$layer[[length(net$layer)]])
   y <- rep(0, nsalidas)
   net <- .Call("ADAPTgdwm_Forward_MLPnet", net, Pvector , new.env(), PACKAGE="AMORE" )
   net <- .Call("ADAPTgdwm_Backwards_MLPnet", net, target, new.env(), PACKAGE="AMORE")   
   return(net)
}
##################################################

#########################################################################################################
ADAPTgdwm.MLPnet.R <- function(net,Pvector,target) {
##########################################################
########################################################## Slower R version of adapt.MLPnet
########################################################## 
   nsalidas <- length(net$layer[[length(net$layer)]])
   y <- rep(0, nsalidas)
   net <- ADAPTgdwm.Forward.MLPnet.R(net,Pvector)
   net <- ADAPTgdwm.Backwards.MLPnet.R(net,target)   
   return(net)
}




