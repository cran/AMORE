

###############################################################################################
restrict <- function (x,x.max,x.min) {
   if (missing(x.max) && missing(x.min)) {
      x.max <- max(x)
      x.min = min(x)
   } 
    x <- -1 + 2 * (x - x.min)/(x.max - x.min)
    return(list(x = x, x.max = x.max, x.min = x.min))
}

###############################################################################################
sim.R.NeuralNet <- function(net,P) {
nmuestras <- nrow(P)
output.layer.neurons <- net$layer[[length(net$layer)]]
nsalidas <- length(output.layer.neurons)
y <- matrix(0, ncol=nsalidas, nrow=nmuestras)
for ( i in 1:nmuestras) {
   net <- forwardpass.R.NeuralNet(net,P[i,])
   for ( j in 1:nsalidas ) {
      y[i,j] <- net$neurons[[ output.layer.neurons[[j]] ]]$v0
   }
}
return(y)
}
##################################################
forwardpass.R.NeuralNet <- function(net,Pvector) {
net$layers$input.layer <- Pvector
for ( ind.layer in 2:length(net$layers) ) {
   for ( ind.neuron in 1:length(net$layers[[ind.layer]]) ) {
       this.neuron <- net$layers[[ind.layer]][[ind.neuron]]
       net$neurons[[this.neuron]] <- forwardpass.R.neuron( net, this.neuron )
   }
}
return(net)
}
##################################################
forwardpass.R.neuron <- function(net,ind.neuron) {
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
   return( neuron )
}
##################################################
sim.C.NeuralNet <- function(net,P) {
nmuestras <- nrow(P)
output.layer.neurons <- net$layer[[length(net$layer)]]
nsalidas <- length(output.layer.neurons)
y <- matrix(0, ncol=nsalidas, nrow=nmuestras)
for ( i in 1:nmuestras) {
   net <- forwardpass.C.NeuralNet(net,P[i,])
   for ( j in 1:nsalidas ) {
      y[i,j] <- net$neurons[[ output.layer.neurons[[j]] ]]$v0
   }
}
return(y)
}
####################################################
forwardpass.C.NeuralNet <- function(net,Pvector) {
net$layers$input.layer <- Pvector
for ( ind.layer in 2:length(net$layers) ) {
   for ( ind.neuron in 1:length(net$layers[[ind.layer]]) ) {
       this.neuron <- net$layers[[ind.layer]][[ind.neuron]]
       net$neurons[[this.neuron]] <- .Call("ForwardPassNeuronC", net, as.integer(this.neuron) , .GlobalEnv,PACKAGE="AMORE")
   }
}
return(net) 
}
##################################################
sim.NeuralNet <- function(net,P) {
nmuestras <- nrow(P)
output.layer.neurons <- net$layer[[length(net$layer)]]
nsalidas <- length(output.layer.neurons)
y <- matrix(0, ncol=nsalidas, nrow=nmuestras)

for ( i in 1:nmuestras) {
   net <- .Call("ForwardPassNeuralNetC", net, P[i,] , .GlobalEnv, PACKAGE="AMORE")
   
   for ( j in 1:nsalidas ) {
      y[i,j] <- net$neurons[[ output.layer.neurons[[j]] ]]$v0
   }
}
return(y)
}
##################################################
## (adaptative)
###
train <- function(net, P, T, n.epochs, g=adapt.NeuralNet, error.criterium="MSE", Stao=NA, report=TRUE, show.step) { 
   epoch.show.step <- 0
   show.step <- show.step -  1
   n.muestras <- nrow(P)

# select error criterium

   if(error.criterium=="MSE") { 
     net$deltaE <- deltaE.MSE
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

# adaptative training
     
   for (epoch in 1:n.epochs) {
      for (i in 1:n.muestras) {
         net <- g(net,P[i,],T[i,])
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
return(net)
}

###############################################################################################
training.report <- function(net,P,T, epoch, error.criterium) {
   P.sim <- sim.NeuralNet(net,P)
#          par(mfrow=c(1,2))
           plot(P,T, col="red", pch="*", ylim=range(rbind(T,P.sim)))
           points(P,P.sim, col="blue", pch="+")
#          plot(P, ideal, col="red", pch=".", ylim=range(rbind(ideal,P.sim)))
#          points(P,P.sim, col="blue", pch=".")
   if(error.criterium=="MSE") { 
           error <- error.MSE(list(prediction=P.sim, target=T))
   } else if(error.criterium=="LMLS") { 
           error <- error.LMLS(list(prediction=P.sim, target=T))
   } else if(error.criterium=="TAO") { 
           error.aux <- error.TAO(list(prediction=P.sim, target=T, other.elements=net$other.elements))
           error     <- error.aux$perf
           net$other.elements$Stao <- error.aux$Stao
           cat("Stao:", net$other.elements$Stao," ")
   }
   cat(paste("Epoch:",epoch,error.criterium,error,"\n", sep=" "))
}
##########################
train.compare <- function(net.start, P, T, ideal=NA, max.epoch, show.step, Stao=1000, criteria=c("MSE","LMLS","TAO")) {

   if (missing(ideal)) {
      ideal <- T
   } 
   training.type  <- pmatch(c("MSE","LMLS","TAO"),criteria)
   T1 <- !is.na(training.type[1])
   T2 <- !is.na(training.type[2])
   T3 <- !is.na(training.type[3])

   if (T1) {  net.1 <- net.start }
   if (T2) {  net.2 <- net.start }
   if (T3) {  net.3 <- net.start }

 for (ntimes in 1:ceiling(max.epoch/show.step)) {
   if (T1) {
      net.1 <- train(net.1, P,T,n.epochs=show.step, g=adapt.NeuralNet, error.criterium="MSE", report=FALSE, show.step=NA)
   }
   if (T2) {
      net.2 <- train(net.2, P,T,n.epochs=show.step, g=adapt.NeuralNet, error.criterium="LMLS",report=FALSE, show.step=NA)
   }
   if (T3) {
      net.3 <- train(net.3, P,T,n.epochs=show.step, g=adapt.NeuralNet, error.criterium="TAO", Stao= Stao, report=FALSE, show.step=Inf)
  }
 
     PlotRange <- range(ideal)
     cat("Epoch:",ntimes*show.step," ")
     if (T1) {
        prediction.1 <- sim.NeuralNet(net.1, P)
        error.1      <- error.MSE(list(prediction=prediction.1, target=ideal))
        cat("MSEMSE: ",error.1,"\t")
        PlotRange    <- range(PlotRange,prediction.1, na.rm=TRUE)
     }
     if (T2) {
        prediction.2 <- sim.NeuralNet(net.2, P)
        error.2      <- error.MSE(list(prediction=prediction.2, target=ideal))
        cat("LMLSMSE:", error.2,"\t") 
        PlotRange    <- range(PlotRange,prediction.2, na.rm=TRUE)
     }
     if (T3) {
        prediction.3  <- sim.NeuralNet(net.3, P)
        error.3       <- error.MSE(list(prediction=prediction.3, target=ideal))
        error.net.tao <- error.TAO(list(prediction=prediction.3, target=ideal, net.3$other.elements))
        Stao <- error.net.tao$Stao
        cat("TAOMSE:", error.3,"Stao:", Stao)
        PlotRange     <- range(PlotRange,prediction.3, na.rm=TRUE)
     }
     cat("\n")

     PlotRange2 <- PlotRange
     PlotRange  <- range(PlotRange2,T)

     par(mfrow=c(1,2))
     plot(P, T, col="black",pch="+", ylim=PlotRange) 
     points(P, ideal, col="brown",pch="+")

     if (T1) { points(P, prediction.1, col="green") }
     if (T2) { points(P, prediction.2, col="blue")  }
     if (T3) { points(P, prediction.3, col="red")   }


     plot(P, ideal, col="black",pch="+", ylim=PlotRange2) 

     if (T1) { points(P, prediction.1, col="green") }
     if (T2) { points(P, prediction.2, col="blue")  }
     if (T3) { points(P, prediction.3, col="red")   }

 }
return(list(net.1, net.2, net.3))
}
###############################################################################################






