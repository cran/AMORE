#include <string.h>

#include <R.h>
#include <Rinternals.h>
#include <Rdefines.h>
#include "AMORE.h"

/******************************************************************************************************************/
/*   MLPnet Level */    /*  Forward and Backwards passes */
/******************************************************************************************************************/
SEXP ADAPTgd_Forward_MLPnet (SEXP net, SEXP Pvector, SEXP rho);
SEXP ADAPTgd_Backwards_MLPnet (SEXP net, SEXP target, SEXP rho);
/******************************************************************************************************************/
/*   MLPneuron Level */
/******************************************************************************************************************/
SEXP ADAPTgd_Forward_MLPneuron(SEXP net, SEXP ind_neuron, SEXP rho);
SEXP ADAPTgd_Backwards_MLPneuron(SEXP net, SEXP ind_neuron, SEXP rho) ;
/******************************************************************************************************************/


/******************************************************************************************************************/
SEXP ADAPTgd_Forward_MLPnet (SEXP net, SEXP Pvector, SEXP rho) {
/******************************************************************************************************************/
int i,ind_layer, ind_neuron;
SEXP this_neuron;

   for ( i=0; i < LENGTH(Pvector); i++) {
      REAL(NET_INPUT)[i] = REAL(Pvector)[i];
   }
   PROTECT(this_neuron=allocVector(INTSXP,1));
   for ( ind_layer=1; ind_layer < LENGTH(NET_LAYERS); ind_layer++ ) {
      for ( ind_neuron=0; ind_neuron < LENGTH( VECTOR_ELT(NET_LAYERS, ind_layer) ) ; ind_neuron++ ) {
         INTEGER(this_neuron)[0] = INTEGER(VECTOR_ELT(NET_LAYERS, ind_layer))[ind_neuron];
         ADAPTgd_Forward_MLPneuron(net, this_neuron, rho);
      }
   }
   UNPROTECT(1);
   return(net);
}
/******************************************************************************************************************/
SEXP ADAPTgd_Backwards_MLPnet (SEXP net, SEXP target, SEXP rho) {
/******************************************************************************************************************/
int i,ind_layer, ind_neuron;
SEXP this_neuron;
   for ( i=0; i < LENGTH(NET_TARGET); i++) {
      REAL(NET_TARGET)[i] = REAL(target)[i];
   }

   PROTECT(this_neuron=allocVector(INTSXP,1));
   for ( ind_layer=-1+LENGTH(NET_LAYERS); ind_layer>0; ind_layer-- ) {
      for ( ind_neuron=-1+LENGTH( VECTOR_ELT(NET_LAYERS, ind_layer) ); ind_neuron >=0;  ind_neuron-- ) {
         INTEGER(this_neuron)[0] = INTEGER(VECTOR_ELT(NET_LAYERS, ind_layer))[ind_neuron];
         ADAPTgd_Backwards_MLPneuron(net, this_neuron, rho );
      }
   }
   UNPROTECT(1);
   return(net);
}

/******************************************************************************************************************/
SEXP ADAPTgd_Forward_MLPneuron(SEXP net, SEXP ind_neuron, SEXP rho) {
/******************************************************************************************************************/
   SEXP neuron, args, R_fcall;
   int ind_weight;
   double x_input, a=0;
   int considered_input;
   PROTECT(neuron=VECTOR_ELT(NET_NEURONS, -1+INTEGER(ind_neuron)[0] ) );
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
       REAL(V1)[0] =  b_tansig / a_tansig * (a_tansig - REAL(V0)[0])*(a_tansig + REAL(V0)[0]);
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
SEXP ADAPTgd_Backwards_MLPneuron(SEXP net, SEXP ind_neuron, SEXP rho) {
/******************************************************************************************************************/
   SEXP neuron,R_fcall, args, arg1, arg2, arg3;
   SEXP aims;
   int ind_weight, ind_other_neuron, that_neuron, that_aim;
   int considered_input;
   int n_protected=0;
   double aux_DELTA ;
   double bias_change;
   double weight_change;
   double x_input;

   PROTECT(neuron=VECTOR_ELT(NET_NEURONS, -1+INTEGER(ind_neuron)[0] ) ); n_protected++;

   if (strcmp(CHAR(STRING_ELT(TYPE,0)),"output")==0) {
      PROTECT(args  = allocVector(VECSXP,3)     ); n_protected++;
   /* PROTECT(arg3  = duplicate(net)            ); n_protected++;      */
      PROTECT(arg3  = net                       ); n_protected++;      
      PROTECT(arg2  = allocVector(REALSXP,1)    ); n_protected++;
      PROTECT(arg1  = allocVector(REALSXP,1)    ); n_protected++;
      REAL(arg1)[0] = REAL(V0)[0];
      REAL(arg2)[0] = REAL(NET_TARGET)[-1+INTEGER(OUTPUT_AIMS)[0]];
      SET_VECTOR_ELT(args, 0, arg1);
      SET_VECTOR_ELT(args, 1, arg2);
      SET_VECTOR_ELT(args, 2, arg3);
      PROTECT(R_fcall = lang2(NET_DELTAE, args) ); n_protected++;
      aux_DELTA = REAL(eval (R_fcall, rho))[0];
   } else {
      aux_DELTA = 0;
      for ( ind_other_neuron=0; ind_other_neuron < LENGTH(OUTPUT_LINKS) ; ind_other_neuron++ ) {
         that_neuron = -1+INTEGER(OUTPUT_LINKS)[ind_other_neuron];
         that_aim    = -1+INTEGER(OUTPUT_AIMS)[ind_other_neuron];
         aux_DELTA  += REAL(VECTOR_ELT(VECTOR_ELT(NET_NEURONS, that_neuron), id_WEIGHTS))[that_aim] * REAL(VECTOR_ELT(VECTOR_ELT(VECTOR_ELT(NET_NEURONS, that_neuron),id_METHOD_DEP_VARIABLES),id_ADAPTgd_DELTA))[0];
       }
   }

   REAL(ADAPTgd_DELTA)[0] = aux_DELTA * REAL(V1)[0];
   bias_change    = -  REAL(ADAPTgd_LEARNING_RATE)[0] * REAL(ADAPTgd_DELTA)[0];
   REAL(BIAS)[0] += bias_change;
   for (ind_weight = 0; ind_weight < LENGTH(WEIGHTS); ind_weight++) {
      considered_input = INTEGER(INPUT_LINKS)[ind_weight];
      if (considered_input < 0 ) {
         x_input = REAL(NET_INPUT)[-1-considered_input];
      } else {
         x_input = REAL(VECTOR_ELT(VECTOR_ELT(NET_NEURONS, -1+considered_input),id_V0))[0];
      }
      weight_change  =  - REAL(ADAPTgd_LEARNING_RATE)[0] * REAL(ADAPTgd_DELTA)[0] * x_input ;
      REAL(WEIGHTS)[ind_weight] += weight_change;
   }
   UNPROTECT(n_protected);
return(neuron);
}
/******************************************************************************************************************/

