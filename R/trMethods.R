##########################################################
#	Adaptative Gradient Descent (without momentum)
##########################################################
ADAPTgd.MLPnet <- function(net, P, T, n.epochs) {
   net <- .Call("ADAPTgd_loop_MLPnet", net, t(P), t(T),as.integer(n.epochs), new.env(), PACKAGE="AMORE" )
   return(net)
}
##################################################


##########################################################
#	Adaptative Gradient Descent (with momentum)
##########################################################
ADAPTgdwm.MLPnet <- function(net,P,T, n.epochs) {
   net <- .Call("ADAPTgdwm_loop_MLPnet", net, t(P), t(T),  as.integer(n.epochs), new.env(), PACKAGE="AMORE" )
   return(net)
}
##################################################


##############################################################
#	BATCHgd ( BATCH gradient descent without momentum )
##############################################################
BATCHgd.MLPnet <- function(net, P, T, n.epochs) { # Each pattern is a row of P, 
#####  First Step: BATCHgd.Forward.MLPnet
   for (ind.MLPneuron in 1:length(net$neurons)) {
      net$neurons[[ind.MLPneuron]]$method.dep.variables$sum.delta.bias <- as.double(0)
      net$neurons[[ind.MLPneuron]]$method.dep.variables$sum.delta.x    <- as.double(numeric(length(net$neurons[[ind.MLPneuron]]$method.dep.variables$sum.delta.x)))
   }
   net <- .Call("BATCHgd_loop_MLPnet", net, t(P), t(T), as.integer(n.epochs), new.env(), PACKAGE="AMORE")
   return(net)
}
##############################################################
#	BATCHgdwm ( BATCH gradient descent with momentum )
##############################################################
BATCHgdwm.MLPnet <- function(net, P, T, n.epochs) { # Each pattern is a row of P, 

##### First step: BATCHgdwm.Forward.MLPnet
   for (ind.MLPneuron in 1:length(net$neurons)) {
      net$neurons[[ind.MLPneuron]]$method.dep.variables$sum.delta.bias <- as.double(0)
      net$neurons[[ind.MLPneuron]]$method.dep.variables$sum.delta.x    <- as.double(numeric(length(net$neurons[[ind.MLPneuron]]$method.dep.variables$sum.delta.x)))
   }
   net <- .Call("BATCHgdwm_loop_MLPnet", net, t(P), t(T), as.integer(n.epochs), new.env(), PACKAGE="AMORE")
   return(net)
}
#######
