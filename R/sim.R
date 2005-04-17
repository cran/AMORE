
###############################################################################################
sim.MLPnet.R <- function(net,P) {
nmuestras <- nrow(P)
output.layer.MLPneurons <- net$layer[[length(net$layer)]]
nsalidas <- length(output.layer.MLPneurons)
y <- matrix(0, ncol=nsalidas, nrow=nmuestras)
for ( i in 1:nmuestras) {
   net <- sim.Forward.MLPnet.R(net,P[i,])
   for ( j in 1:nsalidas ) {
      y[i,j] <- net$neurons[[ output.layer.MLPneurons[[j]] ]]$v0
   }
}
return(y)
}
##################################################
sim.Forward.MLPnet.R <- function(net,Pvector) {
net$input <- Pvector
for ( ind.layer in 2:length(net$layers) ) {
   for ( ind.MLPneuron in 1:length(net$layers[[ind.layer]]) ) {
       this.MLPneuron <- net$layers[[ind.layer]][[ind.MLPneuron]]
       net$neurons[[this.MLPneuron]] <- sim.Forward.MLPneuron( net, this.MLPneuron )
   }
}
return(net)
}
##################################################
sim.Forward.MLPneuron <- function(net,ind.MLPneuron) {
   neuron <-    net$neurons[[ind.MLPneuron]]
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
   return( neuron )
}
##################################################
sim.MLPnet.C <- function(net,P) {
nmuestras <- nrow(P)
output.layer.MLPneurons <- net$layer[[length(net$layer)]]
nsalidas <- length(output.layer.MLPneurons)
y <- matrix(0, ncol=nsalidas, nrow=nmuestras)
for ( i in 1:nmuestras) {
   net <- sim.Forward.MLPnet(net,P[i,])
   for ( j in 1:nsalidas ) {
      y[i,j] <- net$neurons[[ output.layer.MLPneurons[[j]] ]]$v0
   }
}
return(y)
}
####################################################
sim.Forward.MLPnet <- function(net,Pvector) {
net$input <- Pvector
for ( ind.layer in 2:length(net$layers) ) {
   for ( ind.MLPneuron in 1:length(net$layers[[ind.layer]]) ) {
       this.MLPneuron <- net$layers[[ind.layer]][[ind.MLPneuron]]
       net$neurons[[this.MLPneuron]] <- .Call("sim_Forward_MLPNeuron", net, as.integer(this.MLPneuron) , .GlobalEnv,PACKAGE="AMORE")
   }
}
return(net) 
}
##################################################
sim.MLPnet <- function(net,P) {
nmuestras <- nrow(P)
output.layer.MLPneurons <- net$layer[[length(net$layer)]]
nsalidas <- length(output.layer.MLPneurons)
y <- matrix(0, ncol=nsalidas, nrow=nmuestras)

for ( i in 1:nmuestras) {
   net <- .Call("sim_Forward_MLPnet", net, P[i,] , .GlobalEnv, PACKAGE="AMORE")
   for ( j in 1:nsalidas ) {
      y[i,j] <- net$neurons[[ output.layer.MLPneurons[[j]] ]]$v0
   }
}
return(y)
}
###############################################################################################

train <- function(net, P, T, n.epochs, error.criterium="LMS", report=TRUE, show.step, Stao=NA) { 
   epoch.show.step <- 0
   show.step <- show.step -  1
   n.muestras <- nrow(P)
# comprobar method --> adapt or batch
# select error criterium

   if(error.criterium=="LMS") { 
     net$deltaE <- deltaE.LMS
   } else if(error.criterium=="LMLS") { 
     net$deltaE <- deltaE.LMLS
   } else if(error.criterium=="TAO") { 
     if (missing(Stao)) {
        stop("You should enter the value of Stao")
     } else {
        net$deltaE <- deltaE.TAO
        net$other.elements$Stao=Stao
     }
   }

   method <- net$neurons[[1]]$method
   if (method=="ADAPTgd") {
      for (epoch in 1:n.epochs) {
         for (i in 1:n.muestras) {
            net <- ADAPTgd.MLPnet(net,P[i,], T[i,])
         }
         if (report) {
            if (epoch.show.step==show.step) {
              training.report(net, P, T, epoch, error.criterium)
              epoch.show.step <- 0
            } else {
               epoch.show.step <- epoch.show.step + 1 
            }
         }
      }   
   } else if (method=="ADAPTgdwm") {
      for (epoch in 1:n.epochs) {
         for (i in 1:n.muestras) {
            net <- ADAPTgdwm.MLPnet(net,P[i,], T[i,])
         }
         if (report) {
            if (epoch.show.step==show.step) {
              training.report(net, P, T, epoch, error.criterium)
              epoch.show.step <- 0
            } else {
               epoch.show.step <- epoch.show.step + 1 
            }
         }
      }  
   } else if (method=="BATCHgd") {
      for (epoch in 1:n.epochs) {
         net <- BATCHgd.MLPnet(net, P, T)
         if (report) {
            if (epoch.show.step==show.step) {
              training.report(net, P, T, epoch, error.criterium)
              epoch.show.step <- 0
            } else {
               epoch.show.step <- epoch.show.step + 1 
            }
         }
      }  
   } else if (method=="BATCHgdwm") {
      for (epoch in 1:n.epochs) {
         net <- BATCHgdwm.MLPnet(net, P, T)
         if (report) {
            if (epoch.show.step==show.step) {
              training.report(net, P, T, epoch, error.criterium)
              epoch.show.step <- 0
            } else {
               epoch.show.step <- epoch.show.step + 1 
            }
         }
      }  
   }
return(net)
}


###############################################################################################
training.report <- function(net,P,T, epoch, error.criterium) {
   P.sim <- sim.MLPnet(net,P)
#          par(mfrow=c(1,2))
#          plot(P,T, col="red", pch="*", ylim=range(rbind(T,P.sim)))
#          points(P,P.sim, col="blue", pch="+")
#          plot(P, ideal, col="red", pch=".", ylim=range(rbind(ideal,P.sim)))
#          points(P,P.sim, col="blue", pch=".")
   if(error.criterium=="LMS") { 
           error <- error.LMS(list(prediction=P.sim, target=T))
   } else if(error.criterium=="LMLS") { 
           error <- error.LMLS(list(prediction=P.sim, target=T))
   } else if(error.criterium=="TAO") { 
           error.aux <- error.TAO(list(prediction=P.sim, target=T, net=net))
           error     <- error.aux$perf
           net$other.elements$Stao <- error.aux$Stao
           cat("Stao:", net$other.elements$Stao," ")
   }
   cat(paste("Epoch:",epoch,error.criterium,error,"\n", sep=" "))
}





