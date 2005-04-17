#define a_tansig    1.715904708575539
#define b_tansig    0.6666666666666667
#define b_split_a   0.3885219635652736
#define a_sigmoid   1.0

/* ************************************************ */ 
/* net elements */
#define id_NET_LAYERS  0
#define id_NET_NEURONS 1
#define id_NET_INPUT   2
#define id_NET_OUTPUT  3
#define id_NET_TARGET  4
#define id_NET_DELTAE  5
#define id_NET_OTHER_ELEMENTS  4

/**/
#define NET_LAYERS          VECTOR_ELT(net,id_NET_LAYERS)
#define NET_NEURONS         VECTOR_ELT(net,id_NET_NEURONS)
#define NET_INPUT           VECTOR_ELT(net,id_NET_INPUT)
#define NET_OUTPUT          VECTOR_ELT(net,id_NET_OUTPUT)
#define NET_TARGET          VECTOR_ELT(net,id_NET_TARGET)
#define NET_DELTAE          VECTOR_ELT(net,id_NET_DELTAE)
#define NET_OTHER_ELEMENTS  VECTOR_ELT(net,id_NET_OTHER_ELEMENTS)
/* neuron elements */
#define id_ID                    0
#define id_TYPE                  1
#define id_ACTIVATION_FUNCTION   2
#define id_OUTPUT_LINKS          3
#define id_OUTPUT_AIMS           4
#define id_INPUT_LINKS           5
#define id_WEIGHTS               6
#define id_BIAS                  7
#define id_V0                    8
#define id_V1                    9
#define id_F0                    10
#define id_F1                    11
#define id_METHOD                12                  
#define id_METHOD_DEP_VARIABLES  13
/**/
#define ID                       VECTOR_ELT(neuron,id_ID)
#define TYPE                     VECTOR_ELT(neuron,id_TYPE)
#define ACTIVATION_FUNCTION      VECTOR_ELT(neuron,id_ACTIVATION_FUNCTION)
#define OUTPUT_LINKS             VECTOR_ELT(neuron,id_OUTPUT_LINKS)
#define OUTPUT_AIMS              VECTOR_ELT(neuron,id_OUTPUT_AIMS)
#define INPUT_LINKS              VECTOR_ELT(neuron,id_INPUT_LINKS)
#define WEIGHTS                  VECTOR_ELT(neuron,id_WEIGHTS)
#define BIAS                     VECTOR_ELT(neuron,id_BIAS)
#define V0                       VECTOR_ELT(neuron,id_V0)
#define V1                       VECTOR_ELT(neuron,id_V1)
#define F0                       VECTOR_ELT(neuron,id_F0)
#define F1                       VECTOR_ELT(neuron,id_F1)
#define METHOD                   VECTOR_ELT(neuron,id_METHOD)
#define METHOD_DEP_VARIABLES     VECTOR_ELT(neuron,id_METHOD_DEP_VARIABLES)
/* METHOD DEPENDENT VARIABLES */
/* ADAPTgd Adaptative Gradient Descent */
#define id_ADAPTgd_DELTA                   0
#define id_ADAPTgd_LEARNING_RATE           1
/**/
#define ADAPTgd_DELTA          VECTOR_ELT(VECTOR_ELT(neuron,id_METHOD_DEP_VARIABLES),id_ADAPTgd_DELTA)
#define ADAPTgd_LEARNING_RATE  VECTOR_ELT(VECTOR_ELT(neuron,id_METHOD_DEP_VARIABLES),id_ADAPTgd_LEARNING_RATE)
/* ADAPTgdwm Adaptative Gradient Descent with Momentum */
#define id_ADAPTgdwm_DELTA                 0
#define id_ADAPTgdwm_LEARNING_RATE         1
#define id_ADAPTgdwm_MOMENTUM              2
#define id_ADAPTgdwm_FORMER_WEIGHT_CHANGE  3
#define id_ADAPTgdwm_FORMER_BIAS_CHANGE    4
/**/
#define ADAPTgdwm_DELTA                 VECTOR_ELT(VECTOR_ELT(neuron,id_METHOD_DEP_VARIABLES),id_ADAPTgdwm_DELTA)
#define ADAPTgdwm_LEARNING_RATE         VECTOR_ELT(VECTOR_ELT(neuron,id_METHOD_DEP_VARIABLES),id_ADAPTgdwm_LEARNING_RATE)
#define ADAPTgdwm_MOMENTUM              VECTOR_ELT(VECTOR_ELT(neuron,id_METHOD_DEP_VARIABLES),id_ADAPTgdwm_MOMENTUM)
#define ADAPTgdwm_FORMER_WEIGHT_CHANGE  VECTOR_ELT(VECTOR_ELT(neuron,id_METHOD_DEP_VARIABLES),id_ADAPTgdwm_FORMER_WEIGHT_CHANGE)
#define ADAPTgdwm_FORMER_BIAS_CHANGE    VECTOR_ELT(VECTOR_ELT(neuron,id_METHOD_DEP_VARIABLES),id_ADAPTgdwm_FORMER_BIAS_CHANGE)
/* BATCHgd BATCH Gradient Descent */
#define id_BATCHgd_DELTA                   0
#define id_BATCHgd_LEARNING_RATE           1
#define id_BATCHgd_SUM_DELTA_X             2
#define id_BATCHgd_SUM_DELTA_BIAS          3
/**/
#define BATCHgd_DELTA                 VECTOR_ELT(VECTOR_ELT(neuron,id_METHOD_DEP_VARIABLES),id_BATCHgd_DELTA)
#define BATCHgd_LEARNING_RATE         VECTOR_ELT(VECTOR_ELT(neuron,id_METHOD_DEP_VARIABLES),id_BATCHgd_LEARNING_RATE)
#define BATCHgd_SUM_DELTA_X           VECTOR_ELT(VECTOR_ELT(neuron,id_METHOD_DEP_VARIABLES),id_BATCHgd_SUM_DELTA_X)
#define BATCHgd_SUM_DELTA_BIAS        VECTOR_ELT(VECTOR_ELT(neuron,id_METHOD_DEP_VARIABLES),id_BATCHgd_SUM_DELTA_BIAS)
/* BATCHgdwm BATCH Gradient Descent with Momentum */
#define id_BATCHgdwm_DELTA                 0
#define id_BATCHgdwm_LEARNING_RATE         1
#define id_BATCHgdwm_SUM_DELTA_X           2
#define id_BATCHgdwm_SUM_DELTA_BIAS        3
#define id_BATCHgdwm_MOMENTUM              4
#define id_BATCHgdwm_FORMER_WEIGHT_CHANGE  5
#define id_BATCHgdwm_FORMER_BIAS_CHANGE    6
/**/
#define BATCHgdwm_DELTA                 VECTOR_ELT(VECTOR_ELT(neuron,id_METHOD_DEP_VARIABLES),id_BATCHgdwm_DELTA)
#define BATCHgdwm_LEARNING_RATE         VECTOR_ELT(VECTOR_ELT(neuron,id_METHOD_DEP_VARIABLES),id_BATCHgdwm_LEARNING_RATE)
#define BATCHgdwm_SUM_DELTA_X           VECTOR_ELT(VECTOR_ELT(neuron,id_METHOD_DEP_VARIABLES),id_BATCHgdwm_SUM_DELTA_X)
#define BATCHgdwm_SUM_DELTA_BIAS        VECTOR_ELT(VECTOR_ELT(neuron,id_METHOD_DEP_VARIABLES),id_BATCHgdwm_SUM_DELTA_BIAS)
#define BATCHgdwm_MOMENTUM              VECTOR_ELT(VECTOR_ELT(neuron,id_METHOD_DEP_VARIABLES),id_BATCHgdwm_MOMENTUM)
#define BATCHgdwm_FORMER_WEIGHT_CHANGE  VECTOR_ELT(VECTOR_ELT(neuron,id_METHOD_DEP_VARIABLES),id_BATCHgdwm_FORMER_WEIGHT_CHANGE)
#define BATCHgdwm_FORMER_BIAS_CHANGE    VECTOR_ELT(VECTOR_ELT(neuron,id_METHOD_DEP_VARIABLES),id_BATCHgdwm_FORMER_BIAS_CHANGE)


/* OTHER ELEMENTS */
#define id_STAO 0
/**/
#define STAO  VECTOR_ELT(VECTOR_ELT(net,id_OTHER_ELEMENTS),id_STAO)
