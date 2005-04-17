#include <string.h>

#include <R.h>
#include <Rinternals.h>
#include <Rdefines.h>
#include "AMORE.h"
/******************************************************************************************************************/
SEXP sim_Forward_MLPneuron(SEXP net, SEXP ind_neuron, SEXP rho) {
   SEXP neuron, args, R_fcall;
   int ind_weight;
   double x_input, a=0;
   int considered_input;
   PROTECT(neuron=VECTOR_ELT(NET_NEURONS, -1+INTEGER(ind_neuron)[0]) );
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

   UNPROTECT(3);
   return neuron;
}
/******************************************************************************************************************/
SEXP sim_Forward_MLPnet (SEXP net, SEXP Pvector, SEXP rho) {
   int i,ind_layer, ind_neuron;
   SEXP this_neuron;

   for ( i=0; i<LENGTH(Pvector); i++) {
      REAL(NET_INPUT)[i] = REAL(Pvector)[i];
   }
   PROTECT(this_neuron=allocVector(INTSXP,1));
   for ( ind_layer=1; ind_layer < LENGTH(NET_LAYERS); ind_layer++ ) {
      for ( ind_neuron=0; ind_neuron < LENGTH( VECTOR_ELT(NET_LAYERS, ind_layer) ) ; ind_neuron++ ) {
         INTEGER(this_neuron)[0] = INTEGER(VECTOR_ELT(NET_LAYERS, ind_layer))[ind_neuron];
         sim_Forward_MLPneuron(net, this_neuron, rho);
      }
   }
   UNPROTECT(1);
   return(net);
}
/******************************************************************************************************************/

print_MLPneuron (SEXP neuron) {
int i;
   Rprintf("***********************************************************\n");
/* ID */
   Rprintf("ID:\t\t\t%d \n",             INTEGER(ID)[0]            );
/* TYPE */
   Rprintf("TYPE:\t\t\t%s \n",           CHAR(STRING_ELT(TYPE,0))  );
/* ACTIVATION FUNCTION */
   Rprintf("ACT. FUNCTION:\t\t%s\n",     CHAR(STRING_ELT(ACTIVATION_FUNCTION,0)) );
/* OUTPUT LINKS */
   if (INTEGER(OUTPUT_LINKS)[0] != NA_INTEGER ) {
      for (i=0; i<LENGTH(OUTPUT_LINKS); i++) {
         Rprintf("OUTPUT_LINKS %d:\t\t%d \n", i+1, INTEGER(OUTPUT_LINKS)[i]  );
     }
   } else {
      Rprintf("OUTPUT_LINKS:\t\tNA\n");
   }
/* OUTPUT AIMS */
   for (i=0; i<LENGTH(OUTPUT_AIMS); i++) {
      Rprintf("OUTPUT_AIMS.%d:\t\t%d \n", i+1, INTEGER(OUTPUT_AIMS)[i]   );
   }
/* INPUT LINKS */
   for (i=0; i<LENGTH(INPUT_LINKS); i++) {
      Rprintf("INPUT_LINKS.%d:\t\t%d \n", i+1, INTEGER(INPUT_LINKS)[i]  );
   }
/* WEIGHTS */
   for (i=0; i<LENGTH(WEIGHTS); i++) {
      Rprintf("WEIGHTS.%d:\t\t%f \n", i+1, REAL(WEIGHTS)[i]  );
   }
/* BIAS */
   Rprintf("BIAS:\t\t\t%f \n", REAL(BIAS)[0]  );
/* V0 */
   Rprintf("V0:\t\t\t%f \n", REAL(V0)[0]  );
/* V1 */
   Rprintf("V1:\t\t\t%f \n", REAL(V1)[0]  );
/* METHOD */
   Rprintf("METHOD:\t\t\t%s\n", CHAR(STRING_ELT(METHOD,0))  );
   Rprintf("METHOD DEP VARIABLES:\n");
   if (           strcmp(CHAR(STRING_ELT(METHOD,0)),"ADAPTgd"  )==0) {
      /* DELTA */
           Rprintf("DELTA:\t\t\t%f \n",       REAL(ADAPTgd_DELTA)[0]  );
      /* LEARNING RATE */
           Rprintf("LEARNING RATE:\t\t%f \n", REAL(ADAPTgd_LEARNING_RATE)[0]  );
           Rprintf("***********************************************************\n");
   } else    if ( strcmp(CHAR(STRING_ELT(METHOD,0)),"ADAPTgdwm")==0) {
      /* DELTA */
           Rprintf("DELTA:\t\t\t%f \n",       REAL(ADAPTgdwm_DELTA)[0]  );
      /* LEARNING RATE */
           Rprintf("LEARNING RATE:\t\t%f \n", REAL(ADAPTgdwm_LEARNING_RATE)[0]  );
      /* MOMENTUM */
           Rprintf("MOMENTUM:\t\t%f \n",      REAL(ADAPTgdwm_MOMENTUM)[0]  );
      /* FORMER WEIGHT CHANGE */
           for (i=0; i<LENGTH(ADAPTgdwm_FORMER_WEIGHT_CHANGE); i++) {
              Rprintf("FORMER_WEIGHT_CHANGE.%d:\t%f \n", i+1,  REAL(ADAPTgdwm_FORMER_WEIGHT_CHANGE)[i]  );
           }
      /* FORMER BIAS CHANGE */
           Rprintf("FORMER_BIAS_CHANGE:\t%f \n", REAL(ADAPTgdwm_FORMER_BIAS_CHANGE)[0]  );
           Rprintf("***********************************************************\n");
   } else    if ( strcmp(CHAR(STRING_ELT(METHOD,0)),"BATCHgd"  )==0) {
      /* DELTA */
           Rprintf("DELTA:\t\t\t%f \n",       REAL(BATCHgd_DELTA)[0]  );
      /* LEARNING RATE */
           Rprintf("LEARNING RATE:\t\t%f \n", REAL(BATCHgd_LEARNING_RATE)[0]  );
      /* SUM DELTA X */
           for (i=0; i<LENGTH(BATCHgdwm_SUM_DELTA_X); i++) {
              Rprintf("SUM DELTA X %d:\t\t%f \n", i+1,  REAL(BATCHgd_SUM_DELTA_X)[i]  );
           }
      /* SUM DELTA BIAS */
           Rprintf("SUM DELTA BIAS:\t\t%f \n",REAL(BATCHgd_SUM_DELTA_BIAS)[0]  );
           Rprintf("***********************************************************\n");
   } else    if ( strcmp(CHAR(STRING_ELT(METHOD,0)),"BATCHgdwm")==0) {
      /* DELTA */
           Rprintf("DELTA:\t\t\t%f \n",       REAL(BATCHgdwm_DELTA)[0]  );
      /* LEARNING RATE */
           Rprintf("LEARNING RATE:\t\t%f \n", REAL(BATCHgdwm_LEARNING_RATE)[0]  );
      /* MOMENTUM */
           Rprintf("MOMENTUM:\t\t%f \n",      REAL(BATCHgdwm_MOMENTUM)[0]  );
      /* FORMER WEIGHT CHANGE */
           for (i=0; i<LENGTH(ADAPTgdwm_FORMER_WEIGHT_CHANGE); i++) {
              Rprintf("FORMER_WEIGHT_CHANGE.%d:\t%f \n", i+1,  REAL(BATCHgdwm_FORMER_WEIGHT_CHANGE)[i]  );
           }
      /* FORMER BIAS CHANGE */
           Rprintf("FORMER_BIAS_CHANGE:\t%f \n", REAL(BATCHgdwm_FORMER_BIAS_CHANGE)[0]  );
      /* SUM DELTA X */
           for (i=0; i<LENGTH(BATCHgdwm_SUM_DELTA_X); i++) {
              Rprintf("SUM DELTA X %d:\t\t%f \n", i+1,  REAL(BATCHgdwm_SUM_DELTA_X)[i]  );
           }
      /* SUM DELTA BIAS */
           Rprintf("SUM DELTA BIAS:\t\t%f \n",REAL(BATCHgdwm_SUM_DELTA_BIAS)[0]  );
           Rprintf("***********************************************************\n");
   }

}
/******************************************************************************************************************/
