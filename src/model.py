import logging
import keras.backend as K
from keras.layers import Dense, Activation, Embedding, Input
from keras.models import Model
from my_layers import Attention, Average, WeightedSum, WeightedAspectEmb, MaxMargin


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

def create_model(args, kstep, node_size):

    def ortho_reg(weight_matrix):
        ### orthogonal regularization for aspect embedding matrix ###
        w_n = weight_matrix / K.cast(K.epsilon() + K.sqrt(K.sum(K.square(weight_matrix), axis=-1, keepdims=True)), K.floatx())
        reg = K.sum(K.square(K.dot(w_n, K.transpose(w_n)) - K.eye(w_n.shape[0].eval())))
        return args.ortho_reg*reg


    ##### Inputs #####
    sentence_input = Input(shape=(kstep, node_size), dtype='float32', name='sentence_input')
    neg_input = Input(shape=(args.neg_size, kstep, node_size), dtype='float32', name='neg_input')

    print("sentence_input.ndim", sentence_input.ndim)
    print("neg_input.ndim", neg_input.ndim)

    e_w = sentence_input
    y_s = Average()(sentence_input)




    print(y_s.ndim)
    print(e_w.ndim)
    print(K.int_shape(e_w),   K.int_shape(y_s))



    att_weights = Attention(name='att_weights')([e_w, y_s])
    z_s = WeightedSum()([e_w, att_weights])

    print("z_s----------- %d", (z_s.ndim))

    ##### Compute representations of negative instances #####
    #e_neg = word_emb(neg_input)
    e_neg = neg_input
    z_n = Average()(e_neg)


    print("e_neg.ndim")
    print(e_neg.ndim)
    print("z_n.ndim")
    print(z_n.ndim)




    ##### Reconstruction #####
    p_t = Dense(args.aspect_size)(z_s)
    p_t = Activation('softmax', name='p_t')(p_t)
    r_s = WeightedAspectEmb(args.aspect_size, 2405, name='aspect_emb',
            W_regularizer=ortho_reg)(p_t)

    ##### Loss #####

    print("losssssssssssssssssssssssssssssssssssssssssssssssssssssssssss")

    print(K.int_shape(z_s), K.int_shape(r_s))


    loss = MaxMargin(name='max_margin')([z_s, z_n, r_s])
    model = Model(input=[sentence_input, neg_input], output=loss)







    ### Word embedding and aspect embedding initialization ######
    if args.emb_path:
        from w2vEmbReader import W2VEmbReader as EmbReader
        emb_reader = EmbReader(args.emb_path, emb_dim=args.emb_dim)
        logger.info('Initializing word embedding matrix')
        model.get_layer('word_emb').W.set_value(emb_reader.get_emb_matrix_given_vocab(vocab, model.get_layer('word_emb').W.get_value()))
        logger.info('Initializing aspect embedding matrix as centroid of kmean clusters')
        model.get_layer('aspect_emb').W.set_value(emb_reader.get_aspect_matrix(args.aspect_size))

    return model



    





