#include <string.h>

#include <R.h>
#include <Rinternals.h>
#include <Rdefines.h>

#define a_tansig    1.715905
#define b_tansig    0.66666666666666
#define a_sigmoid   1.0

/* ************************************************ */ 
/* net elements */
#define id_LAYERS  0
#define id_NEURONS 1
#define id_TARGET  2
#define id_DELTAE  3
#define id_OTHER_ELEMENTS  4

/**/
#define LAYERS          VECTOR_ELT(net,id_LAYERS)
#define NEURONS         VECTOR_ELT(net,id_NEURONS)
#define TARGET          VECTOR_ELT(net,id_TARGET)
#define DELTAE          VECTOR_ELT(net,id_DELTAE)
#define OTHER_ELEMENTS  VECTOR_ELT(net,id_OTHER_ELEMENTS)
/* layers elements*/
#define id_INPUT_LAYER  0
#define id_HIDDEN_LAYER 1
#define id_OUTPUT_LAYER 2
/**/
#define INPUT_LAYER  VECTOR_ELT(VECTOR_ELT(net,id_LAYERS),id_INPUT_LAYER)
#define HIDDEN_LAYER VECTOR_ELT(VECTOR_ELT(net,id_LAYERS),id_HIDDEN_LAYER)
#define OUTPUT_LAYER VECTOR_ELT(VECTOR_ELT(net,id_LAYERS),id_OUTPUT_LAYER)
/* neuron elements */
#define id_ID                    0
#define id_TYPE                  1
#define id_ACTIVATION_FUNCTION   2
#define id_OUTPUT_LINKS          3
#define id_OUTPUT_AIMS           4
#define id_INPUT_LINKS           5
#define id_WEIGHTS               6
#define id_FORMER_WEIGHT_CHANGE  7
#define id_BIAS                  8
#define id_FORMER_BIAS_CHANGE    9
#define id_V0                    10
#define id_V1                    11
#define id_LEARNING_RATE         12
#define id_SUM_DELTA_X           13
#define id_SUM_DELTA_BIAS        14
#define id_MOMENTUM              15
#define id_F0                    16
#define id_F1                    17
#define id_DELTA                 18
/**/
#define ID                    VECTOR_ELT(neuron,id_ID)
#define TYPE                  VECTOR_ELT(neuron,id_TYPE)
#define ACTIVATION_FUNCTION   VECTOR_ELT(neuron,id_ACTIVATION_FUNCTION)
#define OUTPUT_LINKS          VECTOR_ELT(neuron,id_OUTPUT_LINKS)
#define OUTPUT_AIMS           VECTOR_ELT(neuron,id_OUTPUT_AIMS)
#define INPUT_LINKS           VECTOR_ELT(neuron,id_INPUT_LINKS)
#define WEIGHTS               VECTOR_ELT(neuron,id_WEIGHTS)
#define FORMER_WEIGHT_CHANGE  VECTOR_ELT(neuron,id_FORMER_WEIGHT_CHANGE)
#define BIAS                  VECTOR_ELT(neuron,id_BIAS)
#define FORMER_BIAS_CHANGE    VECTOR_ELT(neuron,id_FORMER_BIAS_CHANGE)
#define V0                    VECTOR_ELT(neuron,id_V0)
#define V1                    VECTOR_ELT(neuron,id_V1)
#define LEARNING_RATE         VECTOR_ELT(neuron,id_LEARNING_RATE)
#define SUM_DELTA_X           VECTOR_ELT(neuron,id_SUM_DELTA_X)
#define SUM_DELTA_BIAS        VECTOR_ELT(neuron,id_SUM_DELTA_BIAS)
#define MOMENTUM              VECTOR_ELT(neuron,id_MOMENTUM)
#define F0                    VECTOR_ELT(neuron,id_F0)
#define F1                    VECTOR_ELT(neuron,id_F1)
#define DELTA                 VECTOR_ELT(neuron,id_DELTA)
/* OTHER ELEMENTS */
#define id_STAO 0
/**/
#define STAO  VECTOR_ELT(VECTOR_ELT(net,id_OTHER_ELEMENTS),id_STAO)

/* ************************************************ */ 
/******************************************************************************************************************/
SEXP ForwardPassNeuronC(SEXP net, SEXP ind_neuron, SEXP rho) {
   SEXP neuron, args, R_fcall;
   int ind_weight;
   double x_input, a=0;
   int considered_input;
   PROTECT(neuron=VECTOR_ELT(NEURONS, -1+INTEGER(ind_neuron)[0]) );
   for (ind_weight=0; ind_weight < LENGTH(WEIGHTS); ind_weight++) {
      considered_input = INTEGER(INPUT_LINKS)[ind_weight];
      if (considered_input < 0 ) {
         x_input = REAL(INPUT_LAYER)[-1-considered_input];
      } else {
         x_input = REAL(VECTOR_ELT(VECTOR_ELT(NEURONS, -1+considered_input),id_V0))[0];
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
SEXP ForwardPassNeuralNetC (SEXP net, SEXP Pvector, SEXP rho) {
int i,ind_layer, ind_neuron;
SEXP this_neuron;

   for ( i=0; i<LENGTH(Pvector); i++) {
      REAL(INPUT_LAYER)[i] = REAL(Pvector)[i];
   }

   PROTECT(this_neuron=allocVector(INTSXP,1));
   for ( ind_layer=1; ind_layer < LENGTH(LAYERS); ind_layer++ ) {
      for ( ind_neuron=0; ind_neuron < LENGTH( VECTOR_ELT(LAYERS, ind_layer) ) ; ind_neuron++ ) {
         INTEGER(this_neuron)[0] = INTEGER(VECTOR_ELT(LAYERS, ind_layer))[ind_neuron];
         ForwardPassNeuronC(net, this_neuron, rho);
      }
   }
   UNPROTECT(1);
   return(net);
}
/******************************************************************************************************************/
SEXP ForwardAdaptNeuronC(SEXP net, SEXP ind_neuron, SEXP rho) {
   SEXP neuron, args, R_fcall;
   int ind_weight;
   double x_input, a=0;
   int considered_input;
   PROTECT(neuron=VECTOR_ELT(NEURONS, -1+INTEGER(ind_neuron)[0] ) );
   for (ind_weight=0; ind_weight < LENGTH(WEIGHTS); ind_weight++) {
      considered_input = INTEGER(INPUT_LINKS)[ind_weight];
      if (considered_input < 0 ) {
         x_input = REAL(INPUT_LAYER)[-1-considered_input];
      } else {
         x_input = REAL(VECTOR_ELT(VECTOR_ELT(NEURONS, -1+considered_input),id_V0))[0];
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
   return neuron;
}
/******************************************************************************************************************/
SEXP ForwardAdaptNeuralNetC (SEXP net, SEXP Pvector, SEXP rho) {
int i,ind_layer, ind_neuron;
SEXP this_neuron;

   for ( i=0; i < LENGTH(Pvector); i++) {
      REAL(INPUT_LAYER)[i] = REAL(Pvector)[i];
   }
   PROTECT(this_neuron=allocVector(INTSXP,1));
   for ( ind_layer=1; ind_layer < LENGTH(LAYERS); ind_layer++ ) {
      for ( ind_neuron=0; ind_neuron < LENGTH( VECTOR_ELT(LAYERS, ind_layer) ) ; ind_neuron++ ) {
         INTEGER(this_neuron)[0] = INTEGER(VECTOR_ELT(LAYERS, ind_layer))[ind_neuron];
         ForwardAdaptNeuronC(net, this_neuron, rho);
      }
   }
   UNPROTECT(1);
   return(net);
}
/******************************************************************************************************************/
print_neuron (SEXP neuron) {
int i;
   Rprintf("***********************************************************\n");
   Rprintf("ID:\t\t\t%d \n",            INTEGER(ID)[0]            );
   Rprintf("TYPE:\t\t\t%s \n",           CHAR(STRING_ELT(TYPE,0))  );
   if (INTEGER(OUTPUT_LINKS)[0] != -1000) {
      for (i=0; i<LENGTH(OUTPUT_LINKS); i++) {
         Rprintf("OUTPUT_LINKS %d:\t\t%d \n", i+1, INTEGER(OUTPUT_LINKS)[i]  );
     }
   } else {
      Rprintf("OUTPUT_LINKS:\t\tNA\n");
   }
   for (i=0; i<LENGTH(OUTPUT_AIMS); i++) {
      Rprintf("OUTPUT_AIMS.%d:\t\t%d \n", i+1, INTEGER(OUTPUT_AIMS)[i]   );
   }
   for (i=0; i<LENGTH(INPUT_LINKS); i++) {
      Rprintf("INPUT_LINKS.%d:\t\t%d \n", i+1, INTEGER(INPUT_LINKS)[i]  );
   }
   for (i=0; i<LENGTH(WEIGHTS); i++) {
      Rprintf("WEIGHTS.%d:\t\t%f \n", i+1, REAL(WEIGHTS)[i]  );
   }
   for (i=0; i<LENGTH(FORMER_WEIGHT_CHANGE); i++) {
      Rprintf("FORMER_WEIGHT_CHANGE.%d:\t%f \n", i+1,  REAL(FORMER_WEIGHT_CHANGE)[i]  );
   }
   Rprintf("BIAS:\t\t\t%f \n", REAL(BIAS)[0]  );
   Rprintf("FORMER_BIAS_CHANGE:\t%f \n", REAL(FORMER_BIAS_CHANGE)[0]  );
   Rprintf("V0:\t\t\t%f \n", REAL(V0)[0]  );
   Rprintf("V1:\t\t\t%f \n", REAL(V1)[0]  );
   Rprintf("LEARNING RATE:\t\t%f \n", REAL(LEARNING_RATE)[0]  );
   for (i=0; i<LENGTH(SUM_DELTA_X); i++) {
      Rprintf("SUM DELTA X %d:\t\t%f \n", i+1,  REAL(SUM_DELTA_X)[i]  );
   }
   Rprintf("SUM DELTA BIAS:\t\t%f \n",REAL(SUM_DELTA_BIAS)[0]  );
   Rprintf("MOMENTUM:\t\t%f \n",      REAL(MOMENTUM)[0]  );
   Rprintf("DELTA:\t\t\t%f \n",       REAL(DELTA)[0]  );
   Rprintf("***********************************************************\n");
}

/******************************************************************************************************************/
SEXP BackpropagateAdaptNeuronC(SEXP net, SEXP ind_neuron, SEXP rho) {
   SEXP neuron,R_fcall, args, arg1, arg2, arg3;

SEXP aims;
   int ind_weight, ind_other_neuron, that_neuron, that_aim;
   int considered_input;
   int n_protected=0;
   double aux_delta ;
   double bias_change;
   double weight_change;
   double x_input;

   PROTECT(neuron=VECTOR_ELT(NEURONS, -1+INTEGER(ind_neuron)[0] ) ); n_protected++;

/*
print_neuron(neuron);
*/


   if (strcmp(CHAR(STRING_ELT(TYPE,0)),"output")==0) {
      PROTECT(args  = allocVector(VECSXP,3)     ); n_protected++;
      PROTECT(arg3  = duplicate(OTHER_ELEMENTS) ); n_protected++;      
      PROTECT(arg2  = allocVector(REALSXP,1)    ); n_protected++;
      PROTECT(arg1  = allocVector(REALSXP,1)    ); n_protected++;
      REAL(arg1)[0] = REAL(V0)[0];
      REAL(arg2)[0] = REAL(TARGET)[-1+INTEGER(OUTPUT_AIMS)[0]];

      SET_VECTOR_ELT(args, 0, arg1);
      SET_VECTOR_ELT(args, 1, arg2);
      SET_VECTOR_ELT(args, 2, arg3);
      
/*
      PROTECT(arg1=allocVector(REALSXP,1)); n_protected++;
      PROTECT(arg2=allocVector(REALSXP,1)); n_protected++;
      REAL(arg1)[0]= REAL(V0)[0];
      REAL(arg2)[0]= REAL(TARGET)[-1+INTEGER(OUTPUT_AIMS)[0]];
      PROTECT(R_fcall = lang3(DELTAE, arg1, arg2) ); n_protected++;        

      Rprintf("arg1  %f \n",  REAL(arg1)[0]);
      Rprintf("arg2  %f \n",  REAL(arg2)[0]);
*/
      PROTECT(R_fcall = lang2(DELTAE, args) ); n_protected++;
      aux_delta = REAL(eval (R_fcall, rho))[0];
/*
      Rprintf("v0 %f \n", REAL(V0)[0]);
      Rprintf("target %f \n", REAL(TARGET)[-1+INTEGER(OUTPUT_AIMS)[0]]);
*/

   } else {
      aux_delta = 0;
      for ( ind_other_neuron=0; ind_other_neuron < LENGTH(OUTPUT_LINKS) ; ind_other_neuron++ ) {
         that_neuron = -1+INTEGER(OUTPUT_LINKS)[ind_other_neuron];
         that_aim    = -1+INTEGER(OUTPUT_AIMS)[ind_other_neuron];
         aux_delta  += REAL(VECTOR_ELT(VECTOR_ELT(NEURONS, that_neuron), id_WEIGHTS))[that_aim] * REAL(VECTOR_ELT(VECTOR_ELT(NEURONS, that_neuron),id_DELTA))[0];
       }
   }

   REAL(DELTA)[0] = aux_delta * REAL(V1)[0];
   bias_change    = REAL(MOMENTUM)[0] * REAL(FORMER_BIAS_CHANGE)[0] -  REAL(LEARNING_RATE)[0] * REAL(DELTA)[0];
   REAL(FORMER_BIAS_CHANGE)[0] = bias_change;
   REAL(BIAS)[0] += bias_change;
/*
Rprintf("Neurona: %d real_bias=%f \n",INTEGER(ind_neuron)[0], REAL(BIAS)[0]);  
*/
   for (ind_weight = 0; ind_weight < LENGTH(WEIGHTS); ind_weight++) {
      considered_input = INTEGER(INPUT_LINKS)[ind_weight];
      if (considered_input < 0 ) {
         x_input = REAL(INPUT_LAYER)[-1-considered_input];
      } else {
         x_input = REAL(VECTOR_ELT(VECTOR_ELT(NEURONS, -1+considered_input),id_V0))[0];
      }
      weight_change  = REAL(MOMENTUM)[0] * REAL(FORMER_WEIGHT_CHANGE)[ind_weight] - REAL(LEARNING_RATE)[0] * REAL(DELTA)[0] * x_input ;
      REAL(WEIGHTS)[ind_weight] += weight_change;
/*
Rprintf("Neurona: %d weight_change=%f \n",INTEGER(ind_neuron)[0], weight_change);
*/
      REAL(FORMER_WEIGHT_CHANGE)[ind_weight] = weight_change;
/*
Rprintf("Neurona: %d real_weights=%f \n",INTEGER(ind_neuron)[0], REAL(WEIGHTS)[ind_weight]);
*/

   }
   UNPROTECT(n_protected);
return(neuron);
}
/******************************************************************************************************************/
/******************************************************************************************************************/
SEXP BackpropagateAdaptNeuralNetC (SEXP net, SEXP target, SEXP rho) {
int i,ind_layer, ind_neuron;
SEXP this_neuron;
   for ( i=0; i < LENGTH(TARGET); i++) {
      REAL(TARGET)[i] = REAL(target)[i];
   }

   PROTECT(this_neuron=allocVector(INTSXP,1));
   for ( ind_layer=-1+LENGTH(LAYERS); ind_layer>0; ind_layer-- ) {
      for ( ind_neuron=-1+LENGTH( VECTOR_ELT(LAYERS, ind_layer) ); ind_neuron >=0;  ind_neuron-- ) {
         INTEGER(this_neuron)[0] = INTEGER(VECTOR_ELT(LAYERS, ind_layer))[ind_neuron];
         BackpropagateAdaptNeuronC(net, this_neuron, rho );
      }
   }
   UNPROTECT(1);
   return(net);
}













/*
R_CallMethodDef callMethods[] = {
{"ForwardPassNeuronC"   , &ForwardPassNeuronC    , 3},
{"ForwardPassNeuralNetC", &ForwardPassNeuralNetC , 2},
{NULL,NULL,0}
};

void R_init_NeuralNetLib(DllInfo *info) {
R_registerRoutines(info, cMethods, callMethods, NULL, NULL);
}
*/
