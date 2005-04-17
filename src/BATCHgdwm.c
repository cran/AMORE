#include <string.h>

#include <R.h>
#include <Rinternals.h>
#include <Rdefines.h>
#include "AMORE.h"

SEXP BATCHgdwm_Forward_MLPneuron(SEXP net, SEXP ind_neuron, SEXP rho);
SEXP BATCHgdwm_ForwardPass_MLPnet (SEXP net, SEXP rho);
SEXP BATCHgdwm_ParcialBackwards_MLPneuron(SEXP net, SEXP ind_neuron, SEXP rho);
SEXP BATCHgdwm_ParcialBackwards_MLPnet (SEXP net, SEXP rho);
SEXP BATCHgdwm_UpdateWeights_MLPnet (SEXP net);
SEXP BATCHgdwm_UpdateWeights_MLPneuron(SEXP net, SEXP ind_neuron);
/******************************************************************************************************************/
/*  MLPnet level */
/******************************************************************************************************************/

/******************************************************************************************************************/
SEXP BATCHgdwm_ForwardPass_MLPnet (SEXP net, SEXP rho) {
/******************************************************************************************************************/
int ind_layer, ind_neuron;
SEXP this_neuron;

   PROTECT(this_neuron=allocVector(INTSXP,1));
   for ( ind_layer=1; ind_layer < LENGTH(NET_LAYERS); ind_layer++ ) {
      for ( ind_neuron=0; ind_neuron < LENGTH( VECTOR_ELT(NET_LAYERS, ind_layer) ) ; ind_neuron++ ) {
         INTEGER(this_neuron)[0] = -1+INTEGER(VECTOR_ELT(NET_LAYERS, ind_layer))[ind_neuron];
         SET_VECTOR_ELT(NET_NEURONS, INTEGER(this_neuron)[0], BATCHgdwm_Forward_MLPneuron(net, this_neuron, rho));
      }
   }
   UNPROTECT(1);
   return(net);
}
/******************************************************************************************************************/
SEXP BATCHgdwm_ParcialBackwards_MLPnet (SEXP net, SEXP rho) {
/******************************************************************************************************************/
   int ind_layer, ind_neuron;
   SEXP this_neuron;

   PROTECT(this_neuron=allocVector(INTSXP,1));
   for ( ind_layer=-1+LENGTH(NET_LAYERS); ind_layer>0; ind_layer-- ) {
      for ( ind_neuron=-1+LENGTH( VECTOR_ELT(NET_LAYERS, ind_layer) ); ind_neuron >=0;  ind_neuron-- ) {
         INTEGER(this_neuron)[0] = -1+INTEGER(VECTOR_ELT(NET_LAYERS, ind_layer))[ind_neuron];
         SET_VECTOR_ELT(NET_NEURONS, INTEGER(this_neuron)[0], BATCHgdwm_ParcialBackwards_MLPneuron(net, this_neuron, rho));
      }
   }
   UNPROTECT(1);
   return(net);
}
/****************************************/
SEXP BATCHgdwm_Forward_MLPneuron(SEXP net, SEXP ind_neuron, SEXP rho) {
/******************************************************************************************************************/
   SEXP neuron, args, R_fcall;
   int ind_weight;
   double x_input, a=0.0;
   int considered_input;
   PROTECT(neuron=VECTOR_ELT(NET_NEURONS, INTEGER(ind_neuron)[0] ) );
   for (ind_weight=0; ind_weight < LENGTH(WEIGHTS); ind_weight++) {
      considered_input = INTEGER(INPUT_LINKS)[ind_weight];
      if (considered_input < 0 ) {
         x_input = REAL(NET_INPUT)[-1-considered_input];
      } else {
         x_input = REAL(VECTOR_ELT(VECTOR_ELT(NET_NEURONS, -1+considered_input),id_V0))[0];
      }
      a +=  REAL(WEIGHTS)[ind_weight] * x_input;
   }
   a += REAL(BIAS)[0];
   PROTECT(args=allocVector(REALSXP,1));   
   REAL(args)[0] = a;
   PROTECT(R_fcall = lang2(F0, args));
   REAL(V0)[0]=REAL(eval (R_fcall, rho))[0];

   if ( strcmp(CHAR(STRING_ELT(ACTIVATION_FUNCTION,0)),"tansig")==0) {
       REAL(V1)[0] =  b_split_a * (a_tansig - REAL(V0)[0])*(a_tansig + REAL(V0)[0]);
   } else if ( strcmp(CHAR(STRING_ELT(ACTIVATION_FUNCTION,0)),"sigmoid")==0) {
       REAL(V1)[0] =  a_sigmoid * REAL(V0)[0] * ( 1 - REAL(V0)[0] );
   } else if ( strcmp(CHAR(STRING_ELT(ACTIVATION_FUNCTION,0)),"purelin")==0) {
       REAL(V1)[0] = 1;
   } else if ( strcmp(CHAR(STRING_ELT(ACTIVATION_FUNCTION,0)),"hardlim")==0) {
       REAL(V1)[0] = NA_REAL;
   } else if ( strcmp(CHAR(STRING_ELT(ACTIVATION_FUNCTION,0)),"custom")==0) {
       PROTECT(args=allocVector(REALSXP,1));   
       REAL(args)[0] = a;
       PROTECT(R_fcall = lang2(F1, args));
       REAL(V1)[0]=REAL(eval (R_fcall, rho))[0];
       UNPROTECT(2);
   }
   UNPROTECT(3);
   return(neuron);
}


/******************************************************************************************************************/
SEXP BATCHgdwm_ParcialBackwards_MLPneuron(SEXP net, SEXP ind_neuron, SEXP rho) {
/******************************************************************************************************************/
   SEXP neuron, R_fcall, args, arg1, arg2, arg3;

   int ind_weight, ind_other_neuron, that_neuron, that_aim;
   int considered_input;
   int n_protected=0;
   double aux_delta ;
   double bias_change;
   double weight_change;
   double x_input;

   PROTECT(neuron=VECTOR_ELT(NET_NEURONS, INTEGER(ind_neuron)[0] ) ); n_protected++;
   if (strcmp(CHAR(STRING_ELT(TYPE,0)),"output")==0) {
      PROTECT(args  = allocVector(VECSXP,3)     ); n_protected++;
  /*  PROTECT(arg3  = duplicate(NET_OTHER_ELEMENTS) ); n_protected++; */
      PROTECT(arg3  = net                       ); n_protected++;
      PROTECT(arg2  = allocVector(REALSXP,1)    ); n_protected++;
      PROTECT(arg1  = allocVector(REALSXP,1)    ); n_protected++;
      REAL(arg1)[0] = REAL(V0)[0];
      REAL(arg2)[0] = REAL(NET_TARGET)[-1+INTEGER(OUTPUT_AIMS)[0]];
      SET_VECTOR_ELT(args, 0, arg1);
      SET_VECTOR_ELT(args, 1, arg2);
      SET_VECTOR_ELT(args, 2, arg3);
/*
      Rprintf("Arg1:\t%f\tArg2:\t%f\tArg3:\t%f\n", REAL(arg1)[0],REAL(arg2)[0],REAL(arg3)[0]);
*/
      PROTECT(R_fcall = lang2(NET_DELTAE, args) ); n_protected++;
      aux_delta = REAL(eval (R_fcall, rho))[0];
        
        /*
	Rprintf("AUX_DELTA:\t%f\n",aux_delta);
	*/

   } else {
      aux_delta = 0.0;
      for ( ind_other_neuron=0; ind_other_neuron < LENGTH(OUTPUT_LINKS) ; ind_other_neuron++ ) {
         that_neuron = -1+INTEGER(OUTPUT_LINKS)[ind_other_neuron];
/*	  
	  Rprintf("that_neuron:\t%d\n",that_neuron);
*/	  
         that_aim    = -1+INTEGER(OUTPUT_AIMS)[ind_other_neuron];
	 
/*
	 Rprintf("that_aim:\t%d\n",that_aim);
	 Rprintf("First:%f\n",REAL(VECTOR_ELT(VECTOR_ELT(NET_NEURONS, that_neuron), id_WEIGHTS))[that_aim]);
	 Rprintf("Second:%f\n",REAL(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(NET_NEURONS, that_neuron),id_METHOD_DEP_VARIABLES),id_BATCHgdwm_DELTA))[0] );
*/
	 	 
         aux_delta  += REAL(VECTOR_ELT(VECTOR_ELT(NET_NEURONS, that_neuron), id_WEIGHTS))[that_aim] * REAL(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(NET_NEURONS, that_neuron),id_METHOD_DEP_VARIABLES),id_BATCHgdwm_DELTA))[0];
/*
	Rprintf("AUX_DELTA:\t%f\n",aux_delta);
*/
       }
   }

   REAL(BATCHgdwm_DELTA)[0] = aux_delta * REAL(V1)[0];
/*
   Rprintf("BATCHgdwm_DELTA:%f\n",REAL(BATCHgdwm_DELTA)[0]);
*/
   for (ind_weight = 0; ind_weight < LENGTH(WEIGHTS); ind_weight++) {
      considered_input = INTEGER(INPUT_LINKS)[ind_weight];
      if (considered_input < 0 ) {
         x_input = REAL(NET_INPUT)[-1-considered_input];
      } else {
         x_input = REAL(VECTOR_ELT(VECTOR_ELT(NET_NEURONS, -1+considered_input),id_V0))[0];
      }
      REAL(BATCHgdwm_SUM_DELTA_X)[ind_weight] += REAL(BATCHgdwm_DELTA)[0] * x_input ;
   }
   REAL(BATCHgdwm_SUM_DELTA_BIAS)[0] += REAL(BATCHgdwm_DELTA)[0];
   UNPROTECT(n_protected);

 /*
 Rprintf("");
 print_MLPneuron (neuron);
*/

return(neuron);
}

/******************************************************************************************************************/
SEXP BATCHgdwm_UpdateWeights_MLPneuron(SEXP net, SEXP ind_neuron) {
/******************************************************************************************************************/
   SEXP neuron;
   int ind_weight;
   double bias_change=0.0, weight_change=0.0;

   PROTECT(neuron=VECTOR_ELT(NET_NEURONS, INTEGER(ind_neuron)[0] ) );

   bias_change    = REAL(BATCHgdwm_MOMENTUM)[0] * REAL(BATCHgdwm_FORMER_BIAS_CHANGE)[0] -  REAL(BATCHgdwm_LEARNING_RATE)[0] * REAL(BATCHgdwm_SUM_DELTA_BIAS)[0];
   REAL(BATCHgdwm_FORMER_BIAS_CHANGE)[0] = bias_change;
   REAL(BIAS)[0] += bias_change;
   for (ind_weight = 0; ind_weight < LENGTH(WEIGHTS); ind_weight++) {
      weight_change  = REAL(BATCHgdwm_MOMENTUM)[0] * REAL(BATCHgdwm_FORMER_WEIGHT_CHANGE)[ind_weight] - REAL(BATCHgdwm_LEARNING_RATE)[0] * REAL(BATCHgdwm_SUM_DELTA_X)[0] ;
      REAL(WEIGHTS)[ind_weight] += weight_change;
      REAL(BATCHgdwm_FORMER_WEIGHT_CHANGE)[ind_weight] = weight_change;
   }
   UNPROTECT(1);
   return(neuron);
}


/******************************************************************************************************************/
SEXP BATCHgdwm_UpdateWeights_MLPnet (SEXP net) {
/******************************************************************************************************************/
int ind_layer, ind_neuron;
SEXP this_neuron;

   PROTECT(this_neuron=allocVector(INTSXP,1));
   for ( ind_layer=1; ind_layer < LENGTH(NET_LAYERS); ind_layer++ ) {
      for ( ind_neuron=0; ind_neuron < LENGTH( VECTOR_ELT(NET_LAYERS, ind_layer) ) ; ind_neuron++ ) {
         INTEGER(this_neuron)[0] = -1+INTEGER(VECTOR_ELT(NET_LAYERS, ind_layer))[ind_neuron];
         SET_VECTOR_ELT(NET_NEURONS, INTEGER(this_neuron)[0], BATCHgdwm_UpdateWeights_MLPneuron(net, this_neuron));
      }
   }
   UNPROTECT(1);
   return(net);
}
/******************************************************************************************************************/

